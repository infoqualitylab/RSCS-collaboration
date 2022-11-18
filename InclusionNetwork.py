# Mark Van Moer, NCSA/RSCS/UIUC

# Show the evolution of an inclusion network over time.
# Time periods are based on when new SRRs appear.

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from math import ceil
from string import ascii_lowercase
import IQLNetwork
from IQLNetwork import read_encoded_csv

class InclusionNetwork(IQLNetwork.IQLNetwork):
    '''Class to encapsulate the inclusion network over its entire history.'''
    def __init__(self):
        super().__init__()

        # periods are subsets of the data based on when new reviews appear. 
        # It will be a list of dictionaries which contain keys for the year 
        # of the current period, the nodes, and the edges visible in that 
        # period.
        self.periods = []
    
    def load_nodes(self):
        # This is an wrapper rather than a pure override because node info is 
        # in two files and we can still use the original one.
        IQLNetwork.IQLNetwork.load_nodes(self)

        # The additional work for this wrapper is to read the review article 
        # details CSV in order to get the search year field.
        reviewdetails = read_encoded_csv(self.reviewdetailscsvpath)

        reviewdetails.columns = reviewdetails.columns.str.strip().str.lower()

        tmp = reviewdetails[[self.id, self.searchyear]]
        # note that after the merge, review articles will have a search_year 
        # that makes sense but included items won't, consequently these will be
        # NaN's and the column type will be float instead of int.
        self.nodes = self.nodes.merge(tmp, how='left')

    def set_aesthetics(self):
        '''Set some per-node aesthetics. Note that networkX drawing funcs 
        only accept per-node values for some node attributes, but not all.
        '''
        # add fill colors
        try:
            conditions = [
                self.nodes[self.kind].eq(self.review),
                self.nodes[self.kind].eq(self.study)
            ]
            choices = [self.review_color, self.study_color]
            self.nodes['fill'] = np.select(conditions, choices, default='black')
        except KeyError:
            self.nodes['fill'] = 'lightgray'

        # add node labels
        # first add a column where the id is a str
        self.nodes['labels'] = self.nodes[self.id].astype('str')
        # now add review label where appropriate
        self.nodes['labels'] = np.where(self.nodes[self.kind] == self.review, 
                self.review_label + self.nodes.labels, self.nodes.labels)

    def node_size_by_degree(self):
        # get node degrees for sizing. nx.degree() returns a DiDegreeView, 
        # which is a wrapper around a dictionary. 
        degsView = nx.degree(self.Graph)
        degsDF = pd.DataFrame.from_dict(degsView)
        degsDF.columns = [self.id, 'degree']
        self.nodes = self.nodes.merge(degsDF)
        self.nodes['degree'] = self.nodes['degree'] * 5

    def _gather_periods(self):
        # NOTE: This method is not intended to be called directly.
        # The idea is that the evolution of the inclusion net is mainly 
        # driven by the arrival of new reviews.
        # New reviews indicate new "periods" (for lack of a better term).
        # Each period contains any new reviews from that year and all
        # studies from appearing since the last period.

        # loop over unique search years grabbing just nodes <= y
        # this is a list of search years as ints
        searchPeriods = self.nodes[self.nodes[self.kind] == self.review][self.searchyear].unique().astype(int)
        searchPeriods = sorted(searchPeriods)

        for y in searchPeriods:
            # grab SRs by search year, PSRs by publication year 
            searchPeriodSRs = self.nodes[(self.nodes[self.kind] == self.review) & (self.nodes[self.searchyear] <= y)]
            searchPeriodPSRs = self.nodes[(self.nodes[self.kind] == self.study) & (self.nodes[self.year] <= y)]

            nodes = pd.concat([searchPeriodSRs,searchPeriodPSRs])
            edges = self.edges[(self.edges['source'].isin(nodes[self.id])) & (self.edges['target'].isin(nodes[self.id]))]
         
            # sources, targets keys are used when drawing period-specific edges
            self.periods.append({'searchyear': y, 
                'nodes': nodes[self.id].tolist(),
                'edges': list(zip(edges['source'].tolist(), edges['target'].tolist())), # list of tuples...
                'sources': edges['source'].tolist(),
                'targets': edges['target'].tolist()})

        # handling any PSRs that happen AFTER the last SRR search year
        # what to put for searchyear is a question... 
        # going with max PSR publication year
        maxPSRyear = max(self.nodes[self.nodes[self.kind] == self.study][self.year])

        if maxPSRyear > searchPeriods[-1]:
            self.periods.append({'searchyear': maxPSRyear,
                'nodes': self.nodes[self.id].tolist(),
                'edges': list(zip(self.edges['source'].tolist(), self.edges['target'].tolist())),
                'sources': self.edges['source'].tolist(),
                'targets': self.edges['target'].tolist()
                })
        

    def _draw_node_subset(self,nodes, kind, shape, edge=None):
        '''
        Used to draw a subset of the nodes. Wrapper around draw_networkx_nodes.

        Nodes have to be drawn as subsets with individiual calls to
        draw_network_x because not all options will take a sequence type, e.g.,
        node shape can only be a string, not a sequence of strings.
        '''
        # kind is review or study
        subnodes = nodes[nodes[self.kind] == kind]

        # convert to dict for networkX
        subnodespos = dict(subnodes[[self.id, 'coords']].values)
        
        # for sizing by degree, change node_size to subnodes['degree'].to_list()
        nx.draw_networkx_nodes(self.Graph,
                subnodespos,
                nodelist=subnodespos.keys(),
                node_color=subnodes['fill'].to_list(),
                node_size=self.node_size,
                node_shape=shape,
                edgecolors=edge)

    def _split_old_new(self,i, period, component='nodes'):
        # distinguish new nodes from old nodes by doing a set difference
        # with the previous period.
        current = period[component]
        previous = self.periods[i-1][component]

        new = list(set(current) - set(previous))
        return previous, new

    def draw(self):
        '''Draws the inclusion network evolution by review "period." Reviews and
        studies new to the respective periods are highlighted in red. 
        '''

        self._gather_periods()

        # if fixed coords, we create and layout the graph based on the entire, 
        # final network.
        coordstr = 'free'
        if self.fixed:
            coordstr = 'fixed'
            self.create_graph()
            self.layout_graph()

        # matplotlib setup for tiled subplots
        if self.tiled:
            fig, axs = plt.subplots(ceil(len(self.periods)/2), 2)
            fig.set_size_inches(16, 23, forward=True)
        else:
            fig, axs = plt.subplots()

        for i, period in enumerate(self.periods):
            # for free-floating drawing, have to create and layout the graph 
            # for each period. The issue is, these are methods, and these 
            # aren't subclasses, though I suppose that would be the more OOP 
            # way to do it. 
            if not self.fixed:
                self.create_graph(period)
                self.layout_graph()

            fig, axs = plt.subplots()

            # this tiles left-to-right, top-to-bottom
            if self.tiled:
                plt.sca(axs[i//2, i%2])

            periodnodesdf = self.nodes[self.nodes[self.id].isin(period['nodes'])]
            periodedgesdf = self.edges[(self.edges['source'].isin(period['sources']) & self.edges['target'].isin(period['targets']))]

            # nodepos contains all the node coords, regardless of kind, and is
            # used to draw edges and node-labels.
            nodepos = dict(periodnodesdf[[self.id, 'coords']].values)

            # set the axes title
            # hack - for Salt but not ExRx, last image/tile needs to have
            # just "year" instead of "search year"
            if self.collection == 'salt' and i == len(self.periods) - 1:
                yearlabel = 'year'
            else:
                yearlabel = 'search year'

            if self.tiled:
                axs[i//2, i%2].set_title('({}) {}: {}'.format(ascii_lowercase[i], yearlabel,
                    period['searchyear'])) 
            else:
                axs.set_title('({}) {}: {}'.format(ascii_lowercase[i], yearlabel,
                    period['searchyear'])) 

            if i == 0 and self.highlight_new:
                # In the first period, there aren't any "new" items in the sense of
                # _split_old_new(), so the highlight color is applied to all SRRs and their
                # connections.
                targets = periodnodesdf[periodnodesdf[self.id].isin(periodedgesdf['target'])]
                nontargets = periodnodesdf[~periodnodesdf[self.id].isin(periodedgesdf['target'])]

                self._draw_node_subset(targets, self.study, self.study_shape, self.new_highlight)
                self._draw_node_subset(nontargets, self.study, self.study_shape)

                self._draw_node_subset(periodnodesdf, self.review, self.review_shape, self.new_highlight)

                nx.draw_networkx_edges(self.Graph, nodepos, periodedgesdf['tuples'].to_list(), 
                        edge_color=self.new_highlight, width=self.edge_width, node_size=self.node_size, arrowsize=self.arrow_size)

            elif i > 0 and self.highlight_new:
                # After the first period, there is a difference between new and old items
                # so only the new items receive the highlighting. This has to be done
                # by splitting because the networkX drawing 
                # review period.

                old_nodes, new_nodes = self._split_old_new(i, period, 'nodes')
                oldperiodnodesdf = self.nodes[self.nodes[self.id].isin(old_nodes)]
                newperiodnodesdf = self.nodes[self.nodes[self.id].isin(new_nodes)]

                self._draw_node_subset(oldperiodnodesdf, self.study, self.study_shape, self.study_color)
                self._draw_node_subset(newperiodnodesdf, self.study, self.study_shape, self.new_highlight)

                self._draw_node_subset(oldperiodnodesdf, self.review, self.review_shape, self.review_color)
                self._draw_node_subset(newperiodnodesdf, self.review, self.review_shape, self.new_highlight)

                old_edges, new_edges = self._split_old_new(i, period, 'edges')
                
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=old_edges, 
                        edge_color=self.edge_color, width=self.edge_width, node_size=self.node_size, arrowsize=self.arrow_size)
                
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=new_edges, 
                        edge_color=self.new_highlight, width=self.edge_width, node_size=self.node_size, arrowsize=self.arrow_size)
            else:
                # Otherwise, don't split on old v. new, i.e., when NOT highlighting new items.
                self._draw_node_subset(periodnodesdf, self.study, self.study_shape)

                self._draw_node_subset(periodnodesdf, self.review, self.review_shape)

                nx.draw_networkx_edges(self.Graph, nodepos, periodedgesdf['tuples'].to_list(), 
                        edge_color=self.edge_color, width=self.edge_width, node_size=self.node_size, arrowsize=self.arrow_size)

            # Draw the labels last
            PSRs = periodnodesdf.loc[periodnodesdf[self.kind] == self.study]
            PSRpos = dict(PSRs[[self.id,'coords']].values)

            nx.draw_networkx_labels(self.Graph, PSRpos,
                    labels = dict(PSRs[[self.id,'labels']].values),
                    font_size=self.study_label_size, font_color=self.study_label_color)
        
            SRs = periodnodesdf.loc[periodnodesdf[self.kind] == self.review]
            SRpos = dict(SRs[[self.id,'coords']].values)

            nx.draw_networkx_labels(self.Graph, SRpos,
                    labels = dict(SRs[[self.id,'labels']].values),
                    font_size=self.review_label_size, font_color=self.review_label_color)

            plt.axis('off')

            # if odd number of subplots, don't draw axes around an empty last 
            # plot in tiled layout.
            if self.tiled and len(self.periods) % 2 == 1:
                axs[-1, -1].axis('off')
            plt.tight_layout()

            if not self.tiled:
                plt.savefig('{}-{}-inclusion-net-{}-{}.png'.format(self.collection,coordstr,self.engine, i), dpi=self.dpi)

            plt.clf()

        if self.tiled:
            plt.savefig('{}-{}-tiled-inclusion-net-{}.png'.format(self.collection,coordstr,self.engine), dpi=self.dpi)
