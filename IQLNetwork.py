# Mark Van Moer, NCSA/RSCS/UIUC

# This is a base class for drawing inclusion networks or
# coauthor networks.

import pandas as pd
import networkx as nx
import yaml
import json
from networkx.drawing.nx_agraph import write_dot

def read_encoded_csv(csvpath):
    '''Helper function for dealing with CSVs saved on different OS's.
    CSV's saved on Windows machines will likely use Windows-1252 code page
    for text encoding. CSV's saved on Linux (and possibly OS X) will 
    likely use utf-8. It all comes down to if a file
    has data outside the Latin 1 charset and in particular, what.
    '''
    encodings = ['utf8', 'cp1252']

    for e in encodings:
        print(f'attempting {e} encoding to read {csvpath}...')
        try:
            df = pd.read_csv(csvpath, encoding=e)
        except UnicodeDecodeError:
            print(f'error using {e}, attempting another encoding...')
        else:
            print(f'file opened with {e} encoding')
            break

    return df

class IQLNetwork:
    '''Base class for both inclusion and coauthor networks.
    Class provides methods for common functionality, e.g., loading data CSVs;
    loading and creating attrs from YAML config files; creating and laying out
    networkX graphs, etc. Drawing and supported functionality is handled
    by the derived classes.
    '''
    def __init__(self):
        self.nodes = None
        self.edges = None
        self.Graph = None

    def load_cfgs(self, cfgpath):
        '''Loads various configuration options from YAML files. Uses setattr
        to make these options class members.
        '''
        print(f'loading configs from {cfgpath}')
        with open(cfgpath, 'r') as cfgfile:
            yaml_cfgs = yaml.load(cfgfile, Loader=yaml.FullLoader)

        # Strings from the CSV which get cleaned below
        csvattrs = {'id', 'year', 'kind', 'review', 'study'}
        
        for k,v in yaml_cfgs.items():
            if k in csvattrs:
                setattr(self, k, v.strip().lower())
            else:
                setattr(self, k, v)

    def load_nodes(self):
        '''Loads node data from nodescsvpath defined in YAML config.'''
        self.nodes = read_encoded_csv(self.nodescsvpath)

        # clean up the column names for consistency
        self.nodes.columns = self.nodes.columns.str.strip().str.lower()
        # strip string column data
        self.nodes = self.nodes.applymap(lambda x: x.strip().lower() \
                if type(x) == str else x)

    def load_edges(self):
        '''Loads edge data from edgescsvpath defined in YAML config.'''
        self.edges = read_encoded_csv(self.edgescsvpath) 

        # This column renaming was originally because other
        # things (Gephi?) assumed 'source' 'target' names for edge endpoints.
        self.edges.columns = self.edges.columns.str.strip().str.lower()
        self.edges = self.edges.rename(
                columns={'citing_id':'source','cited_id':'target'}
                )

        # drop any rows where source or target are NaNs, i.e., in the CSV they
        # are empty. This was needed while the CSVs were in flux. 
        self.edges = self.edges[self.edges['target'].notnull()]
        self.edges = self.edges[self.edges['source'].notnull()]

    def create_graph(self, period=None):
        '''Creates a networkX directed graph for input to layout and 
        drawing algorithms.
        '''
        print('creating graph')

        # for free coordinates, the graph is recreated multiple times
        # so if one already exists it needs to be cleared.
        if self.Graph is not None:
            self.Graph.clear()

        if self.directed:
            self.Graph = nx.DiGraph()
        else:
            self.Graph = nx.Graph()

        if not hasattr(self, 'fixed') or self.fixed:
            # first clause is for Coauthor Networks which are always fixed
            # since they don't implement time at the moment.
            # When drawing fixed coords, create Graph from ALL nodes and edges.
            self.Graph.add_nodes_from(self.nodes[self.id].tolist())
            sources = self.edges['source'].tolist()
            targets = self.edges['target'].tolist()
            self.Graph.add_edges_from(zip(sources, targets))
        else:
            # when drawing free coords, only use nodes and edges from current
            # period.
            self.Graph.add_nodes_from(period['nodes'])
            self.Graph.add_edges_from(zip(period['sources'],period['targets']))

    def layout_graph(self):
        '''Lays out the inclusion network using a pygraphviz algorithm. NetworkX
        interfaces pygraphviz through both/either pydot or agraph. Viable 
        options are for self.engine (defined in YAML):
        neato, dot, twopi, circo, fdp, sfdp
        '''
        print('laying out graph')
        
        # check if coords (and x, y) columns already exist, and drop if so
        # this avoids name clash errors when recreating layout in free coords
        if 'coords' in self.nodes.columns:
            self.nodes = self.nodes.drop(columns=['coords', 'x', 'y'])

        # layout graph and grab coordinates
        fullgraphpos = nx.nx_agraph.graphviz_layout(self.Graph, 
                prog=self.engine)

        # merge the coordinates into the node and edge data frames
        nodecoords = pd.DataFrame.from_dict(fullgraphpos, orient='index', 
            columns=['x', 'y'])
        nodecoords.index.name = self.id
        nodecoords.reset_index(inplace=True)

        # as separate x, y cols
        self.nodes = self.nodes.merge(nodecoords, how='left')

        # a col of (x, y) pairs
        self.nodes['coords'] = list(zip(self.nodes.x, self.nodes.y))

        self.edges = self.edges.merge(nodecoords, how='left', 
                left_on='source', right_on=self.id)
        self.edges = self.edges.drop([self.id],axis=1)
        self.edges = self.edges.merge(nodecoords, how='left',
                left_on='target', right_on=self.id,
                suffixes=tuple(['_source', '_target']))
        self.edges = self.edges.drop([self.id], axis=1)

        # tuples for edgelist
        self.edges['tuples'] = tuple(zip(self.edges.source, self.edges.target))
        # clean up 
        self.edges = self.edges.drop(columns=['x_source', 'x_target', 
            'y_source', 'y_target'])

    def load_layout_json(self):
        '''Loads a layout exported from Gephi as JSON.'''
        with open(self.nodecoordsjson) as json_data:
            data = json.load(json_data)

        nodecoords = pd.DataFrame(data['nodes'])[['x', 'y']]
        nodecoords.index.name = self.id
        nodecoords.reset_index(inplace=True)

        # as separate x, y cols
        self.nodes = pd.merge(self.nodes, nodecoords)

        # a col of (x, y) pairs
        self.nodes['coords'] = list(zip(self.nodes.x, self.nodes.y))

        self.edges = pd.merge(self.edges, nodecoords,
                left_on='source', right_on=self.id)
        self.edges = self.edges.drop([self.id],axis=1)
        self.edges = pd.merge(self.edges, nodecoords,
                left_on='target', right_on=self.id,
                suffixes=tuple(['_source', '_target']))
        self.edges = self.edges.drop([self.id], axis=1)

        # tuples for edgelist
        self.edges['tuples'] = tuple(zip(self.edges.source, self.edges.target))

    def draw(self):
        # overload in subclasses
        pass

    def write_dot(self):
        '''Save self.Graph as a .dot file for external rendering with
        command line GraphViz.
        '''
        write_dot(self.Graph, './{}-network.dot'.format(self.collection))
