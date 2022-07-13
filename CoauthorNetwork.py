# Mark Van Moer, NCSA/RSCS/UIUC

# Draw a coauthor network.

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as colors
import argparse
from math import ceil
from string import ascii_lowercase
import yaml
import json
import Network

class CoauthorNetwork(Network.Network):
    def __init__(self, engine='neato'):
        super().__init__(engine)
        self.node_alpha = 1.0
        self.edge_color='black'
        self.edge_width = 0.5
        self.node_border = 'black'

    def set_node_aesthetics(self):
        self.node_size = 10 
        self.node_color = '#e05b5b'
        self.node_shape = 'o'
        #self.node_border = '#a14242'
        self.node_border = None

    def set_edge_aesthetics(self):
        self.arrowsize = 5

        coauth_min = self.edges['no_of_reports_coauthored'].min()
        coauth_max = self.edges['no_of_reports_coauthored'].max()
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

 
    def draw(self):
        '''Default drawing is of a single static image using self.engine for layout.'''
        print('drawing graph')
        fig, axs = plt.subplots()
        # self.nodes is a pandas dataframe with x, y, and coords cols.
        nodespos = dict(self.nodes[[self._cfgs['id'], 'coords']].values)
        nx.draw_networkx_nodes(self.Graph, nodespos, 
                node_color=self.node_color, 
                node_size=self.node_size, 
                node_shape=self.node_shape, 
                alpha=self.node_alpha,
                edgecolors=self.node_border)

        # for edge_color, want a list of tuples
        nx.draw_networkx_edges(self.Graph, nodespos, 
                arrowsize=self.arrowsize,
                width=self.edge_width, 
                node_size=self.node_size, 
                edge_color=self.edges['rgba'])
        axs.set_title('{} coauthor network'.format(self._cfgs['collection']))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('{}-network-{}.png'.format(self._cfgs['collection'], self.engine), dpi=300)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='draw a static network')
    parser.add_argument('cfgyaml', help='path to a YAML config file')
    args = parser.parse_args()

    network = CoauthorNetwork()
    network.load_cfgs(args.cfgyaml)
    network.load_nodes()
    network.load_edges()
    network.create_graph()
    network.layout_graph()
    network.set_node_aesthetics()
    network.set_edge_aesthetics()
    network.draw()
