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
    def __init__(self):
        super().__init__()

    def set_node_aesthetics(self, const_size=True):
        if not const_size:
            # returns a dictionary of {nodeid:degree} k,v pairs
            degrees = nx.degree(self.Graph)
            self.node_size = [v for k,v in degrees]
        self.nodes['fill'] = 'darkgray'

    def _categorical_fill(self, element):
        # element is self.nodes or self.edges
        # for categorical coloring where percent_srrs is cut into
        # [0.0, 1.0], (0.1, 0.9], (0.9, 1.0]
        element['cut'] = pd.cut(self.nodes['percent_srrs'],
                [0.0, 0.1, 0.9, 1.0], include_lowest=True)
        element['fill'] = None    
        try:
            conditions = [
                    element['cut']==pd.Interval(-0.001,0.1,closed='right'),
                    element['cut']==pd.Interval(0.1,0.9,closed='right'),
                    element['cut']==pd.Interval(0.9,1.0,closed='right')
                    ]
            # colors from Colorbrewer categorical colorblind safe qualitative
            choices = ['#66c2a5','#fc8d62','#8da0cb']
            element['fill'] = np.select(conditions, choices, default='black')
        except KeyError:
            element['fill'] = 'lightgray'

    def set_edge_aesthetics(self):
        self._categorical_fill(self.edges)

        coauth_min = self.edges['no_of_reports_coauthored'].min()
        coauth_max = self.edges['no_of_reports_coauthored'].max()

    def set_edge_aesthetics2(self):
        # this is the original
        # per edge opacity needs to be done as a RGBA tuple ranges [0,1]
        # edge color rather than as alpha parameter in draw_networkx_edges().
        r = [colors.hex2color(self.edge_color)[0]] * len(self.edges)
        g = [colors.hex2color(self.edge_color)[1]] * len(self.edges)
        b = [colors.hex2color(self.edge_color)[2]] * len(self.edges)
        # alpha is just the norm of the no_of_reports_coauthored column
        # but I'm rescaling to [0.1, 1] instead of [0,1] so that there's 
        # always some visibility.
        a = ((self.edges['no_of_reports_coauthored'] - coauth_min) /
                (coauth_max - coauth_min)) * 0.9 + 0.1
        self.edges['rgba'] = tuple(zip(r, g, b, a))

    def _draw_nodes(self, nodespos):
        nx.draw_networkx_nodes(self.Graph,
                nodespos, 
                node_color=self.nodes['fill'],
                node_size=self.node_size, 
                node_shape=self.node_shape, 
                alpha=self.node_alpha,
                linewidths=self.linewidths,
                edgecolors=self.node_border)

    def _draw_cmap_nodes(self, nodespos):
        n = nx.draw_networkx_nodes(self.Graph,
                nodespos,
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
        nx.draw_networkx_edges(self.Graph,
                nodespos, 
                arrowsize=self.arrowsize,
                width=self.edge_width, 
                node_size=self.node_size, 
                edge_color=self.edges['fill'])
    
    def _draw_cmap_edges(self, nodespos):
        nx.draw_networkx_edges(self.Graph,
                nodespos, 
                edge_cmap=self.cmap,
                edge_vmin=self.edges['percent_srrs'].min(),
                edge_vmax=self.edges['percent_srrs'].max(),
                edge_color=self.edges['percent_srrs'],
                arrowsize=self.arrowsize,
                width=self.edge_width, 
                node_size=self.node_size)

    def draw(self):
        '''Default drawing is of a single static image using self.engine for 
        layout.'''
        print('drawing graph')
        self.set_node_aesthetics()
        self.set_edge_aesthetics()

        self.create_graph()
        if self.only_connected:
            self.filter_connected_components()
            connectedstr = 'only-connected'
        else:
            connectedstr = 'entire'
        self.layout_graph()
        

        fig, axs = plt.subplots()
        plt.figure(figsize=(self.figw,self.figh))
        # self.nodes is a pandas dataframe with x, y, and coords cols.
        nodespos = dict(self.nodes[[self.id, 'coords']].values)
        n = None
        if self.usecmap == 'nodes':
            n = self._draw_cmap_nodes(nodespos)
            self._draw_edges(nodespos)
        elif self.usecmap == 'edges':
            self._draw_nodes(nodespos)
            self._draw_cmap_edges(nodespos)
        else:
            self._draw_nodes(nodespos)
            self._draw_edges(nodespos)

        plt.title('{} coauthor network'.format(self.collection))
        plt.axis('off')
        plt.tight_layout()

        # add a colormap legend this works, but the bar is ugly!
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

        plt.savefig('{}-coauthor-network-{}-{}.png'.format(self.collection, 
            self.engine, connectedstr), dpi=300)

    def filter_connected_components(self):
        # just the two largest, not all.
        # nx.connected_components returns a generator of sets
        comps = [_ for _ in sorted(nx.connected_components(self.Graph), 
            key=len, reverse=True)]
        top2comps = comps[:2]
        smallercomps = comps[2:]
        # want to combine those two sets and then filter
        top2compnodes = top2comps[0].union(top2comps[1])
        self.nodes = self.nodes[self.nodes['author_id'].isin(top2compnodes)]

        # also want to remove other nodes from self.Graph 
        smallercompnodes = list(chain.from_iterable(smallercomps))
        self.Graph.remove_nodes_from(smallercompnodes)
