# Mark Van Moer, NCSA/RSCS/UIUC

# Show the evolution of an inclusion network over time.
# Time periods are based on when new Systematic Reviews appear.

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import argparse
from math import ceil
from string import ascii_lowercase
import yaml

class InclusionNetwork:
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
        self.study_shape = 'o'
        self.engine = engine
        self._cfgs = {}
        # SRperiods are subsets of the data based on when new 
        # Systematic Reviews appear. It will be a list of dictionaries
        # which contain keys for the year of the current period,
        # the nodes, and the edges visible in that period.
        self.periods = []
        
    def load_cfgs(self, cfgpath):
        with open(cfgpath, 'r') as cfgfile:
            _tmp_cfgs = yaml.load(cfgfile)
        pathattrs = {'nodescsvpath', 'edgescsvpath'}
        for k,v in _tmp_cfgs.items():
            if k not in pathattrs:
                self._cfgs[k] = v.strip().lower()
            else:
                self._cfgs[k] = v


    def load_nodes(self):
        self.nodes = pd.read_csv(self._cfgs['nodescsvpath'])
        # clean up the column names for consistency
        self.nodes.columns = self.nodes.columns.str.strip().str.lower()
        # strip string column data
        self.nodes = self.nodes.apply(lambda x: x.str.strip() if isinstance(x, str) else x)
    
    def load_edges(self):
        self.edges = pd.read_csv(self._cfgs['edgescsvpath'])
        # MVM - this column renaming was originally because other
        # things (Gephi?) assumed 'source' 'target' names
        self.edges = self.edges.rename(
                columns={'citing_ID':'source','cited_ID':'target'}
                )

    def create_graph(self):
        '''Creates a networkX directed graph for input to layout and 
        drawing algorithms.
        '''
        self.Graph = nx.DiGraph()
        self.Graph.add_nodes_from(self.nodes['id'].tolist())

        # MVM - why am I doing this in a loop instead of
        # through the nx api?
        for idx, row in self.nodes.iterrows():
            self.Graph.add_node(row['id'], **row)

        sources = self.edges['source'].tolist()
        targets = self.edges['target'].tolist()

        self.Graph.add_edges_from(zip(sources, targets))

    def layout_graph(self):
        '''Lays out the inclusion network using a pygraphviz algorithm. NetworkX
        interfaces pygraphviz through both/either pydot or agraph. Viable options are:
        neato, dot, twopi, circo, fdp, sfdp
        '''
        # layout graph and grab coordinates
        fullgraphpos = nx.nx_agraph.graphviz_layout(self.Graph, prog=self.engine)

        # merge the coordinates into the node and edge data frames
        # TODO - remove whatever columns ended up being unused
        nodecoords = pd.DataFrame.from_dict(fullgraphpos, orient='index', 
            columns=['x', 'y'])
        nodecoords.index.name = 'id'
        nodecoords.reset_index(inplace=True)

        # as separate x, y cols
        self.nodes = pd.merge(self.nodes, nodecoords)

        # a col of (x, y) pairs
        self.nodes['coords'] = list(zip(self.nodes.x, self.nodes.y))

        self.edges = pd.merge(self.edges, nodecoords, 
                left_on='source', right_on='id')
        self.edges = self.edges.drop(['id'],axis=1)
        self.edges = pd.merge(self.edges, nodecoords, 
                left_on='target', right_on='id', 
                suffixes=tuple(['_source', '_target']))
        self.edges = self.edges.drop(['id'], axis=1)

        # tuples for edgelist
        self.edges['tuples'] = tuple(zip(self.edges.source, self.edges.target))

    def set_aesthetics(self):
        '''Set some per-node aesthetics. Note that networkX drawing funcs 
        only accept per-node values for some node attributes, but not all.
        '''

        # add fill colors
        conditions = [
            self.nodes['attitude'].eq('inconclusive'),
            self.nodes['attitude'].eq('for'),
            self.nodes['attitude'].eq('against')
        ]

        choices = [self.inconclusive_color, self.for_color, self.against_color]
        self.nodes['fill'] = np.select(conditions, choices, default='black')

        # add node labels
        # first add a column where the id is a str
        self.nodes['labels'] = self.nodes['id'].astype('str')
        # now add 'SR' where appropriate
        self.nodes['labels'] = np.where(self.nodes.type == 'Systematic Review', 
                'SR'+self.nodes.labels, self.nodes.labels)

    def _gather_periods(self):
        # NOTE: This method is not intended to be called directly.
        # The idea is that the evolution of the inclusion net is mainly 
        # driven by the arrival of new Systematic Reviews (SRs).
        # New SRs indicate new "periods" (for lack of a better term).
        # Each period contains any new SRs from that year and all
        # Primary Study Reports (PSRs) from appearing since the last period.

        # NOTE: Pandas is doing all of this with shallow copying so there 
        # aren't duplicates in memory. However, this complicates some of the
        # drawing logic.

        # loop over unique SR years grabbing just nodes <= y
        uniquePeriods = self.nodes[self.nodes['type'] == 'Systematic Review']['year'].unique()
        
        for i, y in enumerate(uniquePeriods):
            nodes = self.nodes[self.nodes['year'] <= y]
            edges = self.edges[self.edges['source'].isin(nodes['id'])]
            maxSR = nodes[nodes['type'] == 'Systematic Review']['id'].max()
            self.periods.append({'endyear': y, 'nodes': nodes, 'edges': edges, 'maxSR':maxSR})


    def draw_graph_evolution(self):
        '''Draws the inclusion network evolution by SR "period." SRs and PSRs
        new to the respective periods are highlighted in red. From the 
        perspective of drawing attributes, there are N subsets of nodes:
        1. new vs old: red outline vs no outline
        2. SR vs PSR: square vs circle shape
        3. Attitude: 3 different fill colors
        for 2x2x3 = 12 possible node aesthetic combinations.
        Fill color is possible to do set prior to drawing because Attitude 
        doesn't change and networkX accepts iterables of fill color.
        
        Edgecolor could also be potentially done prior to drawing if the
        SRperiods were created with deepcopy. Otherwise the problem is that
        different subsets get a partial attribute change which is discouraged
        by pandas best practice.

        Shape however requires splitting because networkX (and matplotlib)
        will only accept a single shape per draw function.
        '''
       
        # MVM: inner func or method?
        # drawing with nx.draw_networkx_{nodes|edges}
        # this way requires that the subsets be dictionaries where the
        # keys are the 'ID' and the values are the coordinate pairs 
        def _draw_sub_nodes(nodes, Type, shape, edge=None):
            # grab the subset of SR vs PSR Type
            subnodes = nodes[nodes['type'] == Type]
            # convert to dict for networkX
            subnodespos = dict(subnodes[['id', 'coords']].values)
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

        # creates the SRperiods list
        self._gather_periods()

        # matplotlib setup for tiled subplots
        fig, axs = plt.subplots(ceil(len(self.periods)/2), 2)
        fig.set_size_inches(8, 11.5, forward=True)

        for i, period in enumerate(self.periods):
            # this tiles left-to-right, top-to-bottom
            plt.sca(axs[i//2, i%2])

            # nodepos contains all the node coords, regardless of type, and is
            # used to draw edges and node-labels.
            nodepos = dict(period['nodes'][['id', 'coords']].values)

            if i > 0:
                # this if case is to only draw the red outlines after the first 
                # SR period.

                # set the axes title
                axs[i//2, i%2].set_title('({}) 2002-{}, with SR1-SR{}'.format(ascii_lowercase[i],
                    period['endyear'],period['maxSR']))

                # split nodes on old vs new 
                old_nodes, new_nodes = _split_old_new(i, period)
                # SRs after PSRs and new after old so they're on top
                _draw_sub_nodes(old_nodes, 'Primary Study Report', self.study_shape)
                _draw_sub_nodes(new_nodes, 'Primary Study Report', self.study_shape, self.new_highlight)
                _draw_sub_nodes(old_nodes, 'Systematic Review', self.review_shape)
                _draw_sub_nodes(new_nodes, 'Systematic Review', self.review_shape, self.new_highlight)

                # split edges on old vs new
                old_edges, new_edges = _split_old_new(i, period, 'edges')

                # MVM: wrap these calls?
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=old_edges['tuples'].to_list(), 
                        edge_color='darkgray', width=self.edge_width, arrowsize=self.arrow_size)
                
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=new_edges['tuples'].to_list(), 
                        edge_color=self.new_highlight, width=self.edge_width, arrowsize=self.arrow_size)
            else:
                axs[i//2, i%2].set_title('(a) 2002, with SR1')

                # first time through, don't split on old v. new
                _draw_sub_nodes(period['nodes'], 'Primary Study Report', self.study_shape)
                _draw_sub_nodes(period['nodes'], 'Systematic Review', self.review_shape)

                nx.draw_networkx_edges(self.Graph, nodepos, period['edges']['tuples'].to_list(), 
                        edge_color='darkgray', width=self.edge_width, arrowsize=self.arrow_size)
        
            # labels are all drawn with the same style regardles of node type
            # so no separation is necessary
            nx.draw_networkx_labels(self.Graph, nodepos, 
                    labels = dict(period['nodes'][['id', 'labels']].values),
                    font_size=6, font_color='#1a1a1a')
            plt.axis('off')
            plt.tight_layout()
        plt.savefig('tiled-inclusion-net-{}.png'.format(self.engine), dpi=300)

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='draw an inclusion network evolution over time')
    parser.add_argument('cfgyaml', help='path to a YAML config file')
    args = parser.parse_args()

    layouts = ['neato', 'dot', 'twopi', 'circo', 'fdp', 'sfdp']
    
    for layout in layouts:
        network = InclusionNetwork(engine=layout)
        network.load_cfgs(args.cfgyaml)
        network.load_nodes()
        network.load_edges()
        network.create_graph()
        network.layout_graph()
        network.set_aesthetics()
        network.draw_graph_evolution()
