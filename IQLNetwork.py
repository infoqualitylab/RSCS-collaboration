# Mark Van Moer, NCSA/RSCS/UIUC

# Draw a graph network using networkX and matplotlib

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import yaml
import json
from networkx.drawing.nx_agraph import write_dot
import sys

class IQLNetwork:
    '''Class to wrap networkX and matplotlib calls.'''
    def __init__(self, engine='neato'):
        self.nodes = None
        self.edges = None
        self.Graph = None
        self.node_size = 50
        self.node_color = '#5f97c2'
        self.node_shape = 'o'
        self.edge_width = 0.5
        self.arrowsize = 5
        self.engine = engine
        self._cfgs = {}
        self._encodings = ['utf8', 'cp1252']

    def load_cfgs(self, cfgpath):
        print(f'loading configs from {cfgpath}')
        with open(cfgpath, 'r') as cfgfile:
            _tmp_cfgs = yaml.load(cfgfile)
        pathattrs = {'nodescsvpath', 'edgescsvpath', 'reviewdetailscsvpath', 'nodecoordsjson'}
        boolattrs = {'tiled', 'loadCoords', 'directed'}
        numericattrs = {'figw', 'figh'}
        for k,v in _tmp_cfgs.items():
            if k in numericattrs:
                self._cfgs[k] = float(v)
            elif k not in pathattrs and k not in boolattrs:
                self._cfgs[k] = v.strip().lower()
            else:
                self._cfgs[k] = v

    def load_nodes(self):
        print('attempting to load nodes from {}'.format(self._cfgs['nodescsvpath']))
        
        # CSV's saved on Windows machines will likely use Windows-1252 code page
        # for text encoding. CSV's saved on Linux (and possibly OS X) will 
        # likely use utf-8. It all comes down to if a file
        # has data outside the Latin 1 charset and in particular, what.
        for e in self._encodings:
            print(f'trying {e} encoding')
            try:
                self.nodes = pd.read_csv(self._cfgs['nodescsvpath'], encoding=e)
            except UnicodeDecodeError:
                print(f'error with {e}, attempting another encoding...')
            else:
                print(f'file opened with {e} encoding')
                break
        # clean up the column names for consistency
        self.nodes.columns = self.nodes.columns.str.strip().str.lower()
        # strip string column data
        self.nodes = self.nodes.applymap(lambda x: x.strip().lower() if type(x) == str else x)

    def load_edges(self):
        print('attempting to load edges from {}'.format(self._cfgs['edgescsvpath']))
       
        
        for e in self._encodings:
            print(f'trying {e} encoding')
            try:
                self.edges = pd.read_csv(self._cfgs['edgescsvpath'])
            except UnicodeDecodeError:
                print(f'error with {e}, attempting another encoding...')
            else:
                print(f'file opened with {e} encoding')
                break 
        # MVM - this column renaming was originally because other
        # things (Gephi?) assumed 'source' 'target' names
        # MVM - change this to a check if 'source' 'target' not there
        # or to a YAML attr
        self.edges.columns = self.edges.columns.str.strip().str.lower()
        self.edges = self.edges.rename(
                columns={'citing_id':'source','cited_id':'target'}
                )

        # drop any rows where source or target are NaNs
        self.edges = self.edges[self.edges['target'].notnull()]
        self.edges = self.edges[self.edges['source'].notnull()]

    def create_graph(self):
        '''Creates a networkX directed graph for input to layout and 
        drawing algorithms.
        '''
        print('creating graph')
        if self._cfgs['directed']:
            self.Graph = nx.DiGraph()
        else:
            self.Graph = nx.Graph()

        self.Graph.add_nodes_from(self.nodes[self._cfgs['id']].tolist())

        # MVM - why am I doing this in a loop instead of
        # through the nx api?
        for idx, row in self.nodes.iterrows():
            self.Graph.add_node(row[self._cfgs['id']], **row)

        sources = self.edges['source'].tolist()
        targets = self.edges['target'].tolist()

        self.Graph.add_edges_from(zip(sources, targets))

    def layout_graph(self):
        '''Lays out the inclusion network using a pygraphviz algorithm. NetworkX
        interfaces pygraphviz through both/either pydot or agraph. Viable options are:
        neato, dot, twopi, circo, fdp, sfdp
        '''
        print('laying out graph')
        # layout graph and grab coordinates
        fullgraphpos = nx.nx_agraph.graphviz_layout(self.Graph, prog=self.engine)

        # merge the coordinates into the node and edge data frames
        # TODO - remove whatever columns ended up being unused
        nodecoords = pd.DataFrame.from_dict(fullgraphpos, orient='index', 
            columns=['x', 'y'])
        nodecoords.index.name = self._cfgs['id']
        nodecoords.reset_index(inplace=True)

        # as separate x, y cols
        self.nodes = pd.merge(self.nodes, nodecoords)

        # a col of (x, y) pairs
        self.nodes['coords'] = list(zip(self.nodes.x, self.nodes.y))

        self.edges = pd.merge(self.edges, nodecoords, 
                left_on='source', right_on=self._cfgs['id'])
        self.edges = self.edges.drop([self._cfgs['id']],axis=1)
        self.edges = pd.merge(self.edges, nodecoords,
                left_on='target', right_on=self._cfgs['id'],
                suffixes=tuple(['_source', '_target']))
        self.edges = self.edges.drop([self._cfgs['id']], axis=1)

        # tuples for edgelist
        self.edges['tuples'] = tuple(zip(self.edges.source, self.edges.target))


    def load_layout_json(self):
        '''Loads a layout exported from Gephi as JSON.'''
        with open(self._cfgs['nodecoordsjson']) as json_data:
            data = json.load(json_data)

        nodecoords = pd.DataFrame(data['nodes'])[['x', 'y']]
        nodecoords.index.name = self._cfgs['id']
        nodecoords.reset_index(inplace=True)

        # as separate x, y cols
        self.nodes = pd.merge(self.nodes, nodecoords)

        # a col of (x, y) pairs
        self.nodes['coords'] = list(zip(self.nodes.x, self.nodes.y))

        self.edges = pd.merge(self.edges, nodecoords,
                left_on='source', right_on=self._cfgs['id'])
        self.edges = self.edges.drop([self._cfgs['id']],axis=1)
        self.edges = pd.merge(self.edges, nodecoords,
                left_on='target', right_on=self._cfgs['id'],
                suffixes=tuple(['_source', '_target']))
        self.edges = self.edges.drop([self._cfgs['id']], axis=1)

        # tuples for edgelist
        self.edges['tuples'] = tuple(zip(self.edges.source, self.edges.target))

    def set_aesthetics(self):
        '''Set some per-node aesthetics. Note that networkX drawing funcs 
        only accept per-node values for some node attributes, but not all.
        '''
        # overload in subclasses
        pass

    def draw(self):
        # overload in subclasses
        pass

    def write_dot(self):
        write_dot(self.Graph, './{}-network.dot'.format(self._cfgs['collection']))

        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='draw a static network')
    parser.add_argument('cfgyaml', help='path to a YAML config file')
    args = parser.parse_args()

    network = Network()
    network.load_cfgs(args.cfgyaml)
    network.load_nodes()
    network.load_edges()
    network.create_graph()
    network.layout_graph()
    network.write_dot()
