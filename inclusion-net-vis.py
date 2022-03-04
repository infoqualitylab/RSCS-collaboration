# plan:
# - read data into node/edges dataframe
# - use a graph algo to layout final, complete inclusion network
# - grab node coords from this layout
# - draw graph evolution directly
#   - loop over years of Study Reviews
#   - draw relevant subset of nodes/edges using previously
#     generated coords

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import figure
import argparse

class InclusionNetwork:
    def __init__(self):
        self.nodes = None
        self.edges = None
        self.Graph = None
        self.SRperiods = []

    def load_nodes(self, nodescsv):
        self.nodes = pd.read_csv(nodescsv)
    
    def load_edges(self, edgescsv):
        self.edges = pd.read_csv(edgescsv)
        # MVM - this column renaming was originally because other
        # things (Gephi?) assumed 'source' 'target' names
        self.edges = self.edges.rename(columns={'citing_ID':'source','cited_ID':'target'})


    def create_graph(self):
        # create Graph object
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

        # layout graph and grab coordinates
        fullgraphpos = nx.nx_agraph.graphviz_layout(self.Graph)

        # merge the coordinates into the node and edge data frames
        nodecoords = pd.DataFrame.from_dict(fullgraphpos, orient='index', 
            columns=['x', 'y'])
        nodecoords.index.name = 'ID'
        nodecoords.reset_index(inplace=True)

        # as separate x, y cols
        self.nodes = pd.merge(self.nodes, nodecoords)

        # a col of (x, y) pairs
        self.nodes['coords'] = list(zip(self.nodes.x, self.nodes.y))

        self.edges = pd.merge(self.edges, nodecoords, left_on='source', right_on='ID')
        self.edges = self.edges.drop(['ID'],axis=1)
        self.edges = pd.merge(self.edges, nodecoords, 
                left_on='target', right_on='ID', 
                suffixes=tuple(['_source', '_target']))
        self.edges = self.edges.drop(['ID'], axis=1)

        # tuples for edgelist
        self.edges['tuples'] = tuple(zip(self.edges.source, self.edges.target))

    def set_aesthetics(self):

        # add fill colors
        conditions = [
            self.nodes['Attitude'].eq('inconclusive'),
            self.nodes['Attitude'].eq('for'),
            self.nodes['Attitude'].eq('against')
        ]

        choices = ['gold', 'lightskyblue', 'lightpink']
        self.nodes['fill'] = np.select(conditions, choices, default='black')

        # add node labels
        # first add a column where the id is a str
        self.nodes['labels'] = self.nodes['ID'].astype('str')
        # now add 'SR' where appropriate
        self.nodes['labels'] = np.where(self.nodes.Type == 'Systematic Review', 
                'SR'+self.nodes.labels, self.nodes.labels)


    def _gather_periods(self):
        # loop over unique SR years grabbing just nodes <= y
        uniquePeriods = self.nodes[self.nodes['Type'] == 'Systematic Review']['year'].unique()

        prev = uniquePeriods[0]
        for i, y in enumerate(uniquePeriods):
            nodes = self.nodes[self.nodes['year'] <= y]
            edges = self.edges[self.edges['source'].isin(nodes['ID'])]
            self.SRperiods.append({'endyear': y, 'nodes': nodes, 'edges': edges})


    def draw_graph_evolution(self):
        self._gather_periods()
        # drawing with nx.draw_networkx_{nodes|edges}
        # this way requires that the subsets be dictionaries where the
        # keys are the 'ID' and the values are the coordinate pairs 
        for i, period in enumerate(self.SRperiods):
            # TODO annoyingly nx.draw turns off padding, margins differently
           
            # SRs are split from PSRs because unlike coloring, shape can only
            # be a single option per draw function.
            SRs = period['nodes'][period['nodes']['Type'] == 'Systematic Review']
            PSRs = period['nodes'][period['nodes']['Type'] == 'Primary Study Report']
            SRnodepos = dict(SRs[['ID', 'coords']].values)
            PSRnodepos = dict(PSRs[['ID', 'coords']].values)
        
            # want to do an edgecolor on nodes that are new, whether SR or PSR,
            # and no edgecolor on older nodes
            # split edges based on if they're new this period, do this with
            # a join on the previous period, i.e., old things are present 
            # in the previous period. pandas doesn't have a direct anti-join, sigh...
            # can't check on year because edges don't have years
            nodepos = dict(period['nodes'][['ID', 'coords']].values)
            if i > 0:
                current_nodes= period['nodes']
                previous_nodes= self.SRperiods[i-1]['nodes']
                tmp = current_nodes.merge(previous_nodes, how='outer', indicator=True)
                new_nodes= tmp[tmp['_merge'] == 'left_only']
                old_nodes= tmp[tmp['_merge'] == 'both']
        
                new_SRs = new_nodes[new_nodes['Type'] == 'Systematic Review']
                old_SRs = old_nodes[old_nodes['Type'] == 'Systematic Review']
        
                new_PSRs = new_nodes[new_nodes['Type'] == 'Primary Study Report']
                old_PSRs = old_nodes[old_nodes['Type'] == 'Primary Study Report']
            
                new_SRs_pos = dict(new_SRs[['ID', 'coords']].values)
                old_SRs_pos = dict(old_SRs[['ID', 'coords']].values)
        
                new_PSRs_pos = dict(new_PSRs[['ID', 'coords']].values)
                old_PSRs_pos = dict(old_PSRs[['ID', 'coords']].values)
        
                nx.draw_networkx_nodes(self.Graph, new_SRs_pos, nodelist=new_SRs_pos.keys(),
                    node_color=new_SRs['fill'].to_list(), node_shape='s', edgecolors='red')
                nx.draw_networkx_nodes(self.Graph, old_SRs_pos, nodelist=old_SRs_pos.keys(),
                    node_color=old_SRs['fill'].to_list(), node_shape='s')
                
                nx.draw_networkx_nodes(self.Graph, new_PSRs_pos, nodelist=new_PSRs_pos.keys(),
                    node_color=new_PSRs['fill'].to_list(), edgecolors='red')
                nx.draw_networkx_nodes(self.Graph, old_PSRs_pos, nodelist=old_PSRs_pos.keys(),
                    node_color=old_PSRs['fill'].to_list())
                
        
                current_edges = period['edges']
                previous_edges = self.SRperiods[i-1]['edges']
                tmp = current_edges.merge(previous_edges, how='outer', indicator=True)
                new_edges = tmp[tmp['_merge'] == 'left_only']
                old_edges = tmp[tmp['_merge'] == 'both']
                
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=old_edges['tuples'].to_list(), 
                        edge_color='darkgray')
                nx.draw_networkx_edges(self.Graph, nodepos, edgelist=new_edges['tuples'].to_list(), 
                        edge_color='red')
            else:
                nx.draw_networkx_nodes(self.Graph, SRnodepos, nodelist=SRnodepos.keys(),
                        node_color=SRs['fill'].to_list(), node_shape='s')
                nx.draw_networkx_nodes(self.Graph, PSRnodepos, nodelist=PSRnodepos.keys(),
                        node_color=PSRs['fill'].to_list())
                nx.draw_networkx_edges(self.Graph, nodepos, period['edges']['tuples'].to_list(), 
                        edge_color='darkgray')
        
            # labels are all drawn with the same style
            # so no separation is necessary
            nx.draw_networkx_labels(self.Graph, nodepos, labels = dict(period['nodes'][['ID', 'labels']].values))
            
        
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('inclusion-net-test-{}.png'.format(i), pad_inches=0, bbox_inches='tight')
            
            plt.close()

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
