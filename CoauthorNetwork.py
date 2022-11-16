# Mark Van Moer, NCSA/RSCS/UIUC

# Draw a coauthor network.

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import IQLNetwork
from itertools import chain

class CoauthorNetwork(IQLNetwork.IQLNetwork):
    def __init__(self, engine='neato'):
        super().__init__(engine)
        self.node_alpha = 1.0
        self.edge_color='black'
        self.edge_width = 1.0
        self.node_border = 'black'
        self.cmap = 'BrBG'

    def set_node_aesthetics(self, const_size=True):
        if const_size:
            self.node_size = 10 
        else:
            # returns a dictionary of {nodeid:degree} k,v pairs
            degrees = nx.degree(self.Graph)
            self.node_size = [v for k,v in degrees]
        self.node_color = '#e05b5b'
        self.node_shape = 'o'
        #self.node_border = '#a14242'
        self.cmap = 'viridis'
        '''
        # for categorical coloring where percent_srrs is cut into
        # [0.0, 1.0], (0.1, 0.9], (0.9, 1.0]
        self.nodes['cut'] = pd.cut(self.nodes['percent_srrs'],
                [0.0, 0.1, 0.9, 1.0], include_lowest=True)
        self.nodes['fill'] = None    
        try:
            conditions = [
                    self.nodes['cut']==pd.Interval(-0.001,0.1,closed='right'),
                    self.nodes['cut']==pd.Interval(0.1,0.9,closed='right'),
                    self.nodes['cut']==pd.Interval(0.9,1.0,closed='right')
                    ]
            # colors from Colorbrewer categorical colorblind safe qualitative
            choices = ['#66c2a5','#fc8d62','#8da0cb']
            self.nodes['fill'] = np.select(conditions, choices, default='black')
        except KeyError:
            self.nodes['fill'] = 'lightgray'
        '''
        self.nodes['fill'] = 'darkgray'

        self.node_border = 'lightgray'
        self.linewidths = 0.5



    def set_edge_aesthetics(self):

        # for categorical coloring where percent_srrs is cut into
        # [0.0, 1.0], (0.1, 0.9], (0.9, 1.0]
        self.edges['cut'] = pd.cut(self.edges['percent_srrs'],
                [0.0, 0.1, 0.9, 1.0], include_lowest=True)
        self.edges['fill'] = None    
        try:
            conditions = [
                    self.edges['cut']==pd.Interval(-0.001,0.1,closed='right'),
                    self.edges['cut']==pd.Interval(0.1,0.9,closed='right'),
                    self.edges['cut']==pd.Interval(0.9,1.0,closed='right')
                    ]
            # colors from Colorbrewer categorical colorblind safe qualitative
            choices = ['#66c2a544','#fc8d62cc','#8da0cbcc']
            self.edges['fill'] = np.select(conditions, choices, default='black')
        except KeyError:
            self.edges['fill'] = 'lightgray'

        coauth_min = self.edges['no_of_reports_coauthored'].min()
        coauth_max = self.edges['no_of_reports_coauthored'].max()

        ''' 
        # this is the original
        # per edge opacity needs to be done as a RGBA tuple ranges [0,1]
        # edge color rather than as alpha parameter in draw_networkx_edges().
        r = [colors.hex2color(self.edge_color)[0]] * len(self.edges)
        g = [colors.hex2color(self.edge_color)[1]] * len(self.edges)
        b = [colors.hex2color(self.edge_color)[2]] * len(self.edges)
        # alpha is just the norm of the no_of_reports_coauthored column
        # but I'm rescaling to [0.1, 1] instead of [0,1] so that there's always some
        # visibility.
        a = ((self.edges['no_of_reports_coauthored'] - coauth_min)/(coauth_max - coauth_min)) * 0.9 + 0.1
        self.edges['rgba'] = tuple(zip(r, g, b, a))
        '''
        # this is for coloring by the cuts

    def _draw_nodes(self, nodespos):
        nx.draw_networkx_nodes(self.Graph, nodespos, 
                #node_color=self.node_color, 
                node_color=self.nodes['fill'],
                node_size=self.node_size, 
                node_shape=self.node_shape, 
                alpha=self.node_alpha,
                linewidths=self.linewidths,
                edgecolors=self.node_border)

    def _draw_cmap_nodes(self, nodespos):
        n = nx.draw_networkx_nodes(self.Graph, nodespos,
                cmap = self.cmap,
                vmin = self.nodes['percent_srrs'].min(),
                vmax = self.nodes['percent_srrs'].max(),
                node_color = self.nodes['percent_srrs'],
                node_shape=self.node_shape,
                node_size=self.node_size,
                alpha=self.node_alpha,
                edgecolors=self.node_border)
        return n

    def _draw_edges(self, nodespos):
        # for edge_color, want a list of tuples
        nx.draw_networkx_edges(self.Graph, nodespos, 
                arrowsize=self.arrowsize,
                width=self.edge_width, 
                node_size=self.node_size, 
                #edge_color=self.edges['rgba'])
                edge_color=self.edges['fill'])
    
    def _draw_cmap_edges(self, nodespos):
        nx.draw_networkx_edges(self.Graph, nodespos, 
                edge_cmap=self.cmap,
                edge_vmin=self.edges['percent_srrs'].min(),
                edge_vmax=self.edges['percent_srrs'].max(),
                edge_color=self.edges['percent_srrs'],
                arrowsize=self.arrowsize,
                width=self.edge_width, 
                node_size=self.node_size)
                


    def draw(self, useCmap=''):
        '''Default drawing is of a single static image using self.engine for layout.'''
        print('drawing graph')
        fig, axs = plt.subplots()
        plt.figure(figsize=(self.figw,self.figh))
        #plt.rcParams['font.size'] = 14
        # self.nodes is a pandas dataframe with x, y, and coords cols.
        nodespos = dict(self.nodes[[self.id, 'coords']].values)
        n = None
        if useCmap == 'nodes':
            n = self._draw_cmap_nodes(nodespos)
            self._draw_edges(nodespos)
        elif useCmap == 'edges':
            self._draw_nodes(nodespos)
            self._draw_cmap_edges(nodespos)
        else:
            self._draw_nodes(nodespos)
            self._draw_edges(nodespos)

        plt.title('{} coauthor network'.format(self.collection))
        plt.axis('off')
        plt.tight_layout()
        # this works, but the bar is ugly!
        '''
        sm = plt.cm.ScalarMappable(cmap=self.cmap)
        sm.set_array([])
        cb = plt.colorbar(
                sm,
                drawedges=False, 
                fraction=0.05, 
                shrink=0.3,
                ticks=None,
                format='%.1f')
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8)
        '''

        #plt.legend(['a', 'b', 'c'])
        plt.savefig('{}-network-{}.png'.format(self.collection, self.engine), dpi=300)

    def filter_connected_components(self):
        # really, just the two largest, not all.
        # nx.connected_components returns a generator of sets
        comps = [_ for _ in sorted(nx.connected_components(self.Graph), key=len, reverse=True)]
        top2comps = comps[:2]
        smallercomps = comps[2:]
        # want to combine those two sets and then filter
        top2compnodes = top2comps[0].union(top2comps[1])
        self.nodes = self.nodes[self.nodes['author_id'].isin(top2compnodes)]

        # also want to remove other nodes from self.Graph 
        smallercompnodes = list(chain.from_iterable(smallercomps))
        self.Graph.remove_nodes_from(smallercompnodes)
