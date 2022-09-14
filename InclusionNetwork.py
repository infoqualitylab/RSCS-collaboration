# Mark Van Moer, NCSA/RSCS/UIUC

# Show the evolution of an inclusion network over time.
# Time periods are based on when new reviews appear.

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import argparse
from math import ceil
from string import ascii_lowercase
import yaml
import json
import IQLNetwork

class InclusionNetwork(IQLNetwork.IQLNetwork):
    '''Class to encapsulate the inclusion network over its entire history.'''
    def __init__(self, engine='neato'):
        super().__init__()
        self.nodes = None
        self.edges = None
        self.Graph = None
        # Here's a thought question, what units are these in? Not pixels...
        self.node_size = 25
        self.edge_width = 0.5
        self.arrow_size = 5
        self.inconclusive_color = '#8da0cb'
        self.for_color = '#66c2a5'
        self.against_color = '#fc8d62'
        self.new_highlight = '#ff3333'
        
        self.review_shape = 's'
        self.review_label = 'SR'
        self.review_color = '#8fb1daff'
        #self.review_edgecolor = '

        self.study_shape = 'o'
        self.study_color = '#dddddd77'
        self.study_edgecolor = '#b889c933'

        self.edge_color = 'lightgray'

        self.engine = engine
        self._cfgs = {}
        # periods are subsets of the data based on when new 
        # reviews appear. It will be a list of dictionaries
        # which contain keys for the year of the current period,
        # the nodes, and the edges visible in that period.
        self.periods = []
    
    def load_nodes(self):
        # This is an wrapper rather than a pure override because node info is in two
        # files and we can still use the original one.
        IQLNetwork.IQLNetwork.load_nodes(self)

        print('attempting to load review article details from: {}'.format(self._cfgs['reviewdetailscsvpath']))
        for e in self._encodings:
            print(f'trying {e} encoding')
            try:
                reviewdetails = pd.read_csv(self._cfgs['reviewdetailscsvpath'], encoding=e)
            except UnicodeDecodeError:
                print(f'error with {e}, attempting another encoding...')
            else:
                print(f'file opened with {e} encoding')
                break

        tmp = reviewdetails[['our_id', 'search_year']]
        # note that after the merge, review articles will have a search_year that makes sense
        # but included items won't, consequently these will be NaN's and the column type
        # will be float instead of int.
        self.nodes = self.nodes.merge(tmp, how='left')

    def set_aesthetics(self):
        '''Set some per-node aesthetics. Note that networkX drawing funcs 
        only accept per-node values for some node attributes, but not all.
        '''

        # add fill colors
        try:
            conditions = [
                self.nodes[self._cfgs['kind']].eq(self._cfgs['review']),
                self.nodes[self._cfgs['kind']].eq(self._cfgs['study'])
            ]
            choices = [self.review_color, self.study_color]
            self.nodes['fill'] = np.select(conditions, choices, default='black')
        except KeyError:
            self.nodes['fill'] = 'lightgray'

        # add node labels
        # first add a column where the id is a str
        self.nodes['labels'] = self.nodes[self._cfgs['id']].astype('str')
        # now add review label where appropriate
        self.nodes['labels'] = np.where(self.nodes[self._cfgs['kind']] == self._cfgs['review'], 
                self.review_label + self.nodes.labels, self.nodes.labels)

    def _gather_periods(self):
        # NOTE: This method is not intended to be called directly.
        # The idea is that the evolution of the inclusion net is mainly 
        # driven by the arrival of new reviews.
        # New reviews indicate new "periods" (for lack of a better term).
        # Each period contains any new reviews from that year and all
        # studies from appearing since the last period.

        # NOTE: Pandas is doing all of this with shallow copying so there 
        # aren't duplicates in memory. However, this complicates some of the
        # drawing logic.

        # loop over unique search years grabbing just nodes <= y
        searchPeriods = self.nodes[self.nodes['our_item_type'] == self._cfgs['review']]['search_year'].unique().astype(int)
        
        for y in sorted(searchPeriods):
            
            searchPeriodSRs = self.nodes[(self.nodes['our_item_type'] == self._cfgs['review']) & (self.nodes['search_year'] <= y)]
            searchPeriodPSRs = self.nodes[(self.nodes['our_item_type'] == self._cfgs['study']) & (self.nodes['publication_year'] <= y)]
            nodes = pd.concat([searchPeriodSRs,searchPeriodPSRs])

            edges = self.edges[(self.edges['source'].isin(nodes[self._cfgs['id']])) & (self.edges['target'].isin(nodes[self._cfgs['id']]))]

            maxReviewId = nodes[nodes[self._cfgs['kind']] == self._cfgs['review']][self._cfgs['id']].max()
            
            self.periods.append({'searchyear': y, 'nodes': nodes, 'edges': edges, 'maxReviewId':maxReviewId})


    def draw(self):
        '''Draws the inclusion network evolution by review "period." Reviews and studies
        new to the respective periods are highlighted in red. From the 
        perspective of drawing attributes, there are N subsets of nodes:
        1. new vs old: red outline vs no outline
        2. review vs study: square vs circle shape
        3. Attitude: 3 different fill colors
        for 2x2x3 = 12 possible node aesthetic combinations.
        Fill color is possible to do set prior to drawing because Attitude 
        doesn't change and networkX accepts iterables of fill color.
        
        Edgecolor could also be potentially done prior to drawing if the
        periods were created with deepcopy. Otherwise the problem is that
        different subsets get a partial attribute change which is discouraged
        by pandas best practice.

        Shape however requires splitting because networkX (and matplotlib)
        will only accept a single shape per draw function.
        '''
       
        # MVM: inner func or method?
        # drawing with nx.draw_networkx_{nodes|edges}
        # this way requires that the subsets be dictionaries where the
        # keys are the 'ID' and the values are the coordinate pairs 
        def _draw_sub_nodes(nodes, kind, shape, edge=None):
            # grab the subset of review vs study kind
            subnodes = nodes[nodes[self._cfgs['kind']] == kind]
            # convert to dict for networkX
            subnodespos = dict(subnodes[[self._cfgs['id'], 'coords']].values)
            nx.draw_networkx_nodes(self.Graph, subnodespos, nodelist=subnodespos.keys(),
                    node_color=subnodes['fill'].to_list(), node_size=self.node_size,
                    node_shape=shape, edgecolors=edge)

        def _split_old_new(i, period, component='nodes'):
            # distinguish new nodes from old nodes by doing an anti-join
            # on the current period vs the previous period. pandas doesn't
            # have a true anti-join function, so do an outer join which
            # retains all values and all rows, but also tag with the
            # indicator parameter so we can compare left-only (new) to
            # both (pre-existing).
            current_nodes= period[component]
            previous_nodes= self.periods[i-1][component]

            tmp = current_nodes.merge(previous_nodes, how='outer', indicator=True)
            old = tmp[tmp['_merge'] == 'both']
            new = tmp[tmp['_merge'] == 'left_only']
            return old, new

        # creates the periods list
        self._gather_periods()

        # matplotlib setup for tiled subplots
        if self._cfgs['tiled']:
            fig, axs = plt.subplots(ceil(len(self.periods)/2), 2)
            fig.set_size_inches(16, 23, forward=True)
        else:
            fig, axs = plt.subplots()

        for i, period in enumerate(self.periods):
            fig, axs = plt.subplots()
            # this tiles left-to-right, top-to-bottom
            
            if self._cfgs['tiled']:
                plt.sca(axs[i//2, i%2])

            # nodepos contains all the node coords, regardless of kind, and is
            # used to draw edges and node-labels.
            nodepos = dict(period['nodes'][[self._cfgs['id'], 'coords']].values)

            # for printing in title
            srslist = period['nodes'][period['nodes']['our_item_type'] == self._cfgs['review']]['our_id'].tolist()

            # set the axes title
            if self._cfgs['tiled']:
                axs[i//2, i%2].set_title('({}) search year: {}'.format(ascii_lowercase[i],
                    period['searchyear'])) 
            else:
                axs.set_title('({}) search year: {}'.format(ascii_lowercase[i],
                    period['searchyear'])) 

            if i > 0:
                # this if case is to only draw the red outlines after the first 
                # review period.

                # split nodes on old vs new 
                old_nodes, new_nodes = _split_old_new(i, period)
                # reviews after studies and new after old so they're on top
                _draw_sub_nodes(old_nodes, self._cfgs['study'], self.study_shape, self.study_color)
                _draw_sub_nodes(new_nodes, self._cfgs['study'], self.study_shape, self.new_highlight)
                PSRs = period['nodes'].loc[period['nodes']['our_item_type'] == 'includeditem']
                PSRpos = dict(PSRs[[self._cfgs['id'],'coords']].values)
                nx.draw_networkx_labels(self.Graph, PSRpos,
                        labels = dict(PSRs[[self._cfgs['id'],'labels']].values),
                        font_size=4, font_color='#1a1a1aaa')
                _draw_sub_nodes(old_nodes, self._cfgs['review'], self.review_shape, self.review_color)
                _draw_sub_nodes(new_nodes, self._cfgs['review'], self.review_shape, self.new_highlight)

                # split edges on old vs new
                old_edges, new_edges = _split_old_new(i, period, 'edges')

                # MVM: wrap these calls?
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=old_edges['tuples'].to_list(), 
                        edge_color=self.edge_color, width=self.edge_width, node_size=self.node_size, arrowsize=5)
                
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=new_edges['tuples'].to_list(), 
                        edge_color=self.new_highlight, width=self.edge_width, node_size=self.node_size, arrowsize=5)
            else:

                # first time through, don't split on old v. new
                _draw_sub_nodes(period['nodes'], self._cfgs['study'], self.study_shape)
                PSRs = period['nodes'].loc[period['nodes']['our_item_type'] == 'includeditem']
                PSRpos = dict(PSRs[[self._cfgs['id'],'coords']].values)
                nx.draw_networkx_labels(self.Graph, PSRpos,
                        labels = dict(PSRs[[self._cfgs['id'],'labels']].values),
                        font_size=4, font_color='#1a1a1aaa')
                _draw_sub_nodes(period['nodes'], self._cfgs['review'], self.review_shape)

                nx.draw_networkx_edges(self.Graph, nodepos, period['edges']['tuples'].to_list(), 
                        edge_color=self.edge_color, width=self.edge_width, node_size=self.node_size, arrowsize=5)
        
            # labels are all drawn with the same style regardles of node kind 
            # so no separation is necessary
            '''
            nx.draw_networkx_labels(self.Graph, nodepos, 
                    labels = dict(period['nodes'][[self._cfgs['id'], 'labels']].values),
                    font_size=6, font_color='#1a1a1a')

            '''

            SRs = period['nodes'].loc[period['nodes']['our_item_type'] == 'reviewarticle']
            SRpos = dict(SRs[[self._cfgs['id'],'coords']].values)
            nx.draw_networkx_labels(self.Graph, SRpos,
                    labels = dict(SRs[[self._cfgs['id'],'labels']].values),
                    font_size=6, font_color='#1a1a1a')

            

            plt.axis('off')

            # if odd number of subplots, don't draw axes around an empty last plot
            if self._cfgs['tiled'] and len(self.periods) % 2 == 1:
                axs[-1, -1].axis('off')
            plt.tight_layout()
            if not self._cfgs['tiled']:
                plt.savefig('{}-inclusion-net-{}-{}.png'.format(self._cfgs['collection'],self.engine, i), dpi=300)

            plt.clf()
            #plt.cla()
            #plt.close()

        if self._cfgs['tiled']:
            plt.savefig('{}-tiled-inclusion-net-{}.png'.format(self._cfgs['collection'],self.engine), dpi=300)


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='draw an inclusion network evolution over time')
    parser.add_argument('cfgyaml', help='path to a YAML config file')
    args = parser.parse_args()
    
    network = InclusionNetwork()
    network.load_cfgs(args.cfgyaml)
    network.load_nodes()
    network.load_edges()
    network.create_graph()
    network.layout_graph()
    network.set_aesthetics()
    network.draw()
