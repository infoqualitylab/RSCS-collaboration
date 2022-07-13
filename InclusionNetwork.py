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
import Network

class InclusionNetwork(Network.Network):
    '''Class to encapsulate the inclusion network over its entire history.'''
    def __init__(self, engine='neato'):
        self.nodes = None
        self.edges = None
        self.Graph = None
        # Here's a thought question, what units are these in? Not pixels...
        self.node_size = 100
        self.edge_width = 0.5
        self.arrow_size = 5
        self.inconclusive_color = '#8da0cb'
        self.for_color = '#66c2a5'
        self.against_color = '#fc8d62'
        self.new_highlight = '#ff3333'
        self.review_shape = 's'
        self.review_label = 'SR'
        self.study_shape = 'o'
        self.engine = engine
        self._cfgs = {}
        # periods are subsets of the data based on when new 
        # reviews appear. It will be a list of dictionaries
        # which contain keys for the year of the current period,
        # the nodes, and the edges visible in that period.
        self.periods = []
        
    def set_aesthetics(self):
        '''Set some per-node aesthetics. Note that networkX drawing funcs 
        only accept per-node values for some node attributes, but not all.
        '''

        # add fill colors
        try:
            conditions = [
                self.nodes[self._cfgs['attitude']].eq('inconclusive'),
                self.nodes[self._cfgs['attitude']].eq('for'),
                self.nodes[self._cfgs['attitude']].eq('against')
            ]
            choices = [self.inconclusive_color, self.for_color, self.against_color]
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

        # loop over unique review years grabbing just nodes <= y
        uniquePeriods = self.nodes[self.nodes[self._cfgs['kind']] == self._cfgs['review']][self._cfgs['year']].unique()
        
        for i, y in enumerate(uniquePeriods):
            nodes = self.nodes[self.nodes[self._cfgs['year']] <= y]
            edges = self.edges[self.edges['source'].isin(nodes[self._cfgs['id']])]
            maxReviewId = nodes[nodes[self._cfgs['kind']] == self._cfgs['review']][self._cfgs['id']].max()
            startyear = nodes[nodes[self._cfgs['kind']] == self._cfgs['review']][self._cfgs['year']].min()
            self.periods.append({'endyear': y, 'nodes': nodes, 'edges': edges, 'maxReviewId':maxReviewId, 
                'startyear': startyear})


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
            # this tiles left-to-right, top-to-bottom
            
            if self._cfgs['tiled']:
                plt.sca(axs[i//2, i%2])

            # nodepos contains all the node coords, regardless of kind, and is
            # used to draw edges and node-labels.
            nodepos = dict(period['nodes'][[self._cfgs['id'], 'coords']].values)

            if i > 0:
                # this if case is to only draw the red outlines after the first 
                # review period.

                # set the axes title
                if self._cfgs['tiled']:
                    axs[i//2, i%2].set_title('({0}) {1}-{2}, with {3}1-{3}{4}'.format(ascii_lowercase[i],
                        period['startyear'], period['endyear'], self.review_label, period['maxReviewId']))
                else:
                    axs.set_title('({0}) {1}-{2}, with {3}1-{3}{4}'.format(ascii_lowercase[i],
                        period['startyear'], period['endyear'], self.review_label, period['maxReviewId']))
                # split nodes on old vs new 
                old_nodes, new_nodes = _split_old_new(i, period)
                # reviews after studies and new after old so they're on top
                _draw_sub_nodes(old_nodes, self._cfgs['study'], self.study_shape)
                _draw_sub_nodes(new_nodes, self._cfgs['study'], self.study_shape, self.new_highlight)
                _draw_sub_nodes(old_nodes, self._cfgs['review'], self.review_shape)
                _draw_sub_nodes(new_nodes, self._cfgs['review'], self.review_shape, self.new_highlight)

                # split edges on old vs new
                old_edges, new_edges = _split_old_new(i, period, 'edges')

                # MVM: wrap these calls?
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=old_edges['tuples'].to_list(), 
                        edge_color='darkgray', width=self.edge_width, arrowsize=self.arrow_size)
                
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=new_edges['tuples'].to_list(), 
                        edge_color=self.new_highlight, width=self.edge_width, arrowsize=self.arrow_size)
            else:
                if self._cfgs['tiled']:
                    if self._cfgs['collection'] == 'salt':
                        axs[i//2, i%2].set_title('(a) {}, with {}1'.format(period['startyear'],self.review_label))
                    else:
                        axs[i//2, i%2].set_title('({0}) {1}-{2}, with {3}1-{3}{4}'.format(ascii_lowercase[i],
                            period['startyear'], period['endyear'], self.review_label, period['maxReviewId']))
                else:
                    axs.set_title('(a) {}, with {}1'.format(period['startyear'],self.review_label))

                # first time through, don't split on old v. new
                _draw_sub_nodes(period['nodes'], self._cfgs['study'], self.study_shape)
                _draw_sub_nodes(period['nodes'], self._cfgs['review'], self.review_shape)

                nx.draw_networkx_edges(self.Graph, nodepos, period['edges']['tuples'].to_list(), 
                        edge_color='darkgray', width=self.edge_width, arrowsize=self.arrow_size)
        
            # labels are all drawn with the same style regardles of node kind 
            # so no separation is necessary
            nx.draw_networkx_labels(self.Graph, nodepos, 
                    labels = dict(period['nodes'][[self._cfgs['id'], 'labels']].values),
                    font_size=6, font_color='#1a1a1a')
            plt.axis('off')

            # if odd number of subplots, don't draw axes around an empty last plot
            if self._cfgs['tiled'] and len(self.periods) % 2 == 1:
                axs[-1, -1].axis('off')
            plt.tight_layout()
            if not self._cfgs['tiled']:
                plt.savefig('{}-inclusion-net-{}-{}.png'.format(self._cfgs['collection'],self.engine, i), dpi=300)
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
