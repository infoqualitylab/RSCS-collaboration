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

class InclusionNetwork:
    '''Class to encapsulate the inclusion network over its entire history.'''
    def __init__(self):
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
        # SRperiods are subsets of the data based on when new 
        # Systematic Reviews appear. It will be a list of dictionaries
        # which contain keys for the year of the current period,
        # the nodes, and the edges visible in that period.
        self.SRperiods = []

    def load_nodes(self, nodescsv):
        self.nodes = pd.read_csv(nodescsv)
        # MVM - there was one Attitude which had a trailing space
        self.nodes.Attitude = self.nodes.Attitude.str.strip()
    
    def load_edges(self, edgescsv):
        self.edges = pd.read_csv(edgescsv)
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
        self.Graph.add_nodes_from(self.nodes['ID'].tolist())

        # MVM - why am I doing this in a loop instead of
        # through the nx api?
        for idx, row in self.nodes.iterrows():
            self.Graph.add_node(row['ID'], **row)

        sources = self.edges['source'].tolist()
        targets = self.edges['target'].tolist()

        self.Graph.add_edges_from(zip(sources, targets))

    def layout_graph(self):
        '''Lays out the inclusion network using the default AGraph
        Graphviz algorithm.
        '''
        #TODO - allow different layout options
        # layout graph and grab coordinates
        fullgraphpos = nx.nx_agraph.graphviz_layout(self.Graph)

        # merge the coordinates into the node and edge data frames
        # TODO - remove whatever columns ended up being unused
        nodecoords = pd.DataFrame.from_dict(fullgraphpos, orient='index', 
            columns=['x', 'y'])
        nodecoords.index.name = 'ID'
        nodecoords.reset_index(inplace=True)

        # as separate x, y cols
        self.nodes = pd.merge(self.nodes, nodecoords)

        # a col of (x, y) pairs
        self.nodes['coords'] = list(zip(self.nodes.x, self.nodes.y))

        self.edges = pd.merge(self.edges, nodecoords, 
                left_on='source', right_on='ID')
        self.edges = self.edges.drop(['ID'],axis=1)
        self.edges = pd.merge(self.edges, nodecoords, 
                left_on='target', right_on='ID', 
                suffixes=tuple(['_source', '_target']))
        self.edges = self.edges.drop(['ID'], axis=1)

        # tuples for edgelist
        self.edges['tuples'] = tuple(zip(self.edges.source, self.edges.target))

    def set_aesthetics(self):
        '''Set some per-node aesthetics. Note that networkX drawing funcs 
        only accept per-node values for some node attributes, but not all.
        '''
        # add fill colors
        conditions = [
            self.nodes['Attitude'].eq('inconclusive'),
            self.nodes['Attitude'].eq('for'),
            self.nodes['Attitude'].eq('against')
        ]

        choices = [self.inconclusive_color, self.for_color, self.against_color]
        self.nodes['fill'] = np.select(conditions, choices, default='black')

        # add node labels
        # first add a column where the id is a str
        self.nodes['labels'] = self.nodes['ID'].astype('str')
        # now add 'SR' where appropriate
        self.nodes['labels'] = np.where(self.nodes.Type == 'Systematic Review', 
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
        uniquePeriods = self.nodes[self.nodes['Type'] == 'Systematic Review']['year'].unique()
        
        for i, y in enumerate(uniquePeriods):
            nodes = self.nodes[self.nodes['year'] <= y]
            edges = self.edges[self.edges['source'].isin(nodes['ID'])]
            maxSR = nodes[nodes['Type'] == 'Systematic Review']['ID'].max()
            self.SRperiods.append({'endyear': y, 'nodes': nodes, 'edges': edges, 'maxSR':maxSR})


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

        self._gather_periods()
        fig, axs = plt.subplots(ceil(len(self.SRperiods)/2), 2)
        fig.set_size_inches(8, 11.5, forward=True)
        # drawing with nx.draw_networkx_{nodes|edges}
        # this way requires that the subsets be dictionaries where the
        # keys are the 'ID' and the values are the coordinate pairs 

        for i, period in enumerate(self.SRperiods):
            # this tiles left-right, top-bottom
            plt.sca(axs[i//2, i%2])
            # nodepos contains all the node coords, regardless of type, and is
            # used to draw edges and node-labels.
            nodepos = dict(period['nodes'][['ID', 'coords']].values)
            if i > 0:
                # this if case is to only draw the red outlines after the first 
                # SR period.

                axs[i//2, i%2].set_title('({}) 2002-{}, with SR1-SR{}'.format(ascii_lowercase[i],
                    period['endyear'],period['maxSR']))

                # distinguish new nodes from old nodes by doing an anti-join
                # on the current period vs the previous period. pandas doesn't
                # have a true anti-join function, so do an outer join which
                # retains all values and all rows, but also tag with the
                # indicator parameter so we can compare left-only (new) to
                # both (pre-existing).
                current_nodes= period['nodes']
                previous_nodes= self.SRperiods[i-1]['nodes']

                tmp = current_nodes.merge(previous_nodes, how='outer', indicator=True)
                new_nodes= tmp[tmp['_merge'] == 'left_only']
                old_nodes= tmp[tmp['_merge'] == 'both']

                old_SRs = old_nodes[old_nodes['Type'] == 'Systematic Review']
                old_PSRs = old_nodes[old_nodes['Type'] == 'Primary Study Report']
                old_SRs_pos = dict(old_SRs[['ID', 'coords']].values)
                old_PSRs_pos = dict(old_PSRs[['ID', 'coords']].values)
       
                # draw the old SRs without an outline
                nx.draw_networkx_nodes(self.Graph, old_SRs_pos, nodelist=old_SRs_pos.keys(),
                    node_color=old_SRs['fill'].to_list(), node_size=self.node_size, node_shape='s')

                # draw the old PSRs witout an outline
                nx.draw_networkx_nodes(self.Graph, old_PSRs_pos, nodelist=old_PSRs_pos.keys(),
                    node_color=old_PSRs['fill'].to_list(), node_size=self.node_size)
              
                new_SRs = new_nodes[new_nodes['Type'] == 'Systematic Review']
                new_PSRs = new_nodes[new_nodes['Type'] == 'Primary Study Report']
                # Convert to dict for how networkX expects the data
                new_SRs_pos = dict(new_SRs[['ID', 'coords']].values)
                new_PSRs_pos = dict(new_PSRs[['ID', 'coords']].values)

                # Draw new items second so they overlay old ones
                # draw the new PSRs with a red outline
                nx.draw_networkx_nodes(self.Graph, new_PSRs_pos, nodelist=new_PSRs_pos.keys(),
                    node_color=new_PSRs['fill'].to_list(), node_size=self.node_size, edgecolors=self.new_highlight)

                # draw the new SRs with a red outline
                nx.draw_networkx_nodes(self.Graph, new_SRs_pos, nodelist=new_SRs_pos.keys(),
                    node_color=new_SRs['fill'].to_list(), node_shape='s', node_size=self.node_size, edgecolors=self.new_highlight)
                
                # Same process, but now for the edges
                current_edges = period['edges']
                previous_edges = self.SRperiods[i-1]['edges']
                tmp = current_edges.merge(previous_edges, how='outer', indicator=True)
                new_edges = tmp[tmp['_merge'] == 'left_only']
                old_edges = tmp[tmp['_merge'] == 'both']
                               
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=old_edges['tuples'].to_list(), 
                        edge_color='darkgray', width=self.edge_width, arrowsize=self.arrow_size)
                
                # draw the new edges second so that the red overlaps the darkgray
                # of the old edges.
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=new_edges['tuples'].to_list(), 
                        edge_color=self.new_highlight, width=self.edge_width, arrowsize=self.arrow_size)
            else:
                axs[i//2, i%2].set_title('(a) 2002, with SR1')
                # for the first SR period, draw without any outlining.
                SRs = period['nodes'][period['nodes']['Type'] == 'Systematic Review']
                PSRs = period['nodes'][period['nodes']['Type'] == 'Primary Study Report']
                SRnodepos = dict(SRs[['ID', 'coords']].values)
                PSRnodepos = dict(PSRs[['ID', 'coords']].values)

                nx.draw_networkx_nodes(self.Graph, SRnodepos, nodelist=SRnodepos.keys(),
                        node_color=SRs['fill'].to_list(), node_shape='s', node_size=self.node_size)
                nx.draw_networkx_nodes(self.Graph, PSRnodepos, nodelist=PSRnodepos.keys(),
                        node_color=PSRs['fill'].to_list(), node_size=self.node_size)

                nx.draw_networkx_edges(self.Graph, nodepos, period['edges']['tuples'].to_list(), 
                        edge_color='darkgray', width=self.edge_width, arrowsize=self.arrow_size)
        
            # labels are all drawn with the same style regardles of node type
            # so no separation is necessary
            nx.draw_networkx_labels(self.Graph, nodepos, 
                    labels = dict(period['nodes'][['ID', 'labels']].values),
                    font_size=6, font_color='#1a1a1a')
            plt.axis('off')
            plt.tight_layout()
        plt.savefig('tiled-inclusion-net.png', dpi=300)

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='draw an inclusion network evolution over time')
    parser.add_argument('nodescsv', help='path to a CSV containing nodes')
    parser.add_argument('edgescsv', help='path to a CSV containing edges')
    args = parser.parse_args()

    saltnetwork = InclusionNetwork()
    saltnetwork.load_nodes(args.nodescsv)
    saltnetwork.load_edges(args.edgescsv)
    saltnetwork.create_graph()
    saltnetwork.layout_graph()
    saltnetwork.set_aesthetics()
    saltnetwork.draw_graph_evolution()
