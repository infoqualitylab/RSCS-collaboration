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

datapath = '~/Vis/projects/schneider-rscs/data/derived/'
nodesdf = pd.read_csv(datapath + 'article_attr.mvm-edit.csv')

max_year = nodesdf['year'].max()
nodesdf['age'] = max_year - nodesdf['year']
ages = sorted(nodesdf['age'].unique())

# create Graph object
G = nx.DiGraph()
G.add_nodes_from(nodesdf['id'].tolist())

for idx, row in nodesdf.iterrows():
    G.add_node(row['id'], **row)

edgesdf = pd.read_csv(datapath + 'inclusion_net_edges.mvm-edit.csv')
sources = edgesdf['source'].tolist()
targets = edgesdf['target'].tolist()

G.add_edges_from(zip(sources, targets))

# layout graph and grab coordinates
fullgraphpos = nx.nx_agraph.graphviz_layout(G)
nx.draw(G, fullgraphpos)
plt.savefig('backwards-fullgraph.png')
plt.close()

# merge the coordinates
nodecoords = pd.DataFrame.from_dict(fullgraphpos, orient='index', 
        columns=['x', 'y'])
nodecoords.index.name = 'id'
nodecoords.reset_index(inplace=True)

# as separate x, y cols
nodesdf = pd.merge(nodesdf, nodecoords)

# a col of (x, y) pairs
nodesdf['coords'] = list(zip(nodesdf.x, nodesdf.y))

# add fill colors
conditions = [
        nodesdf['Attitude'].eq('inconclusive'),
        nodesdf['Attitude'].eq('for'),
        nodesdf['Attitude'].eq('against')
]

choices = ['gold', 'lightskyblue', 'lightpink']
nodesdf['fill'] = np.select(conditions, choices, default='black')

# add node labels
# first add a column where the id is a str
nodesdf['labels'] = nodesdf['id'].astype('str')
# now add 'SR' where appropriate
nodesdf['labels'] = np.where(nodesdf.Type == 'Systematic Review', 'SR'+nodesdf.labels, nodesdf.labels)

edgesdf = pd.merge(edgesdf, nodecoords, left_on='source', right_on='id')
edgesdf = edgesdf.drop(['id'],axis=1)
edgesdf = pd.merge(edgesdf, nodecoords, 
        left_on='target', right_on='id', 
        suffixes=tuple(['_source', '_target']))
edgesdf = edgesdf.drop(['id'], axis=1)

# tuples for edgelist
edgesdf['tuples'] = tuple(zip(edgesdf.source, edgesdf.target))

# get the unique SR years to make cohorts
#SRs = nodesdf[nodesdf['Type'] == 'Systematic Review']

# loop over unique SR years grabbing just nodes <= y
SRperiods = []
#uniquePeriods = SRs['year'].unique()
uniquePeriods = nodesdf[nodesdf['Type'] == 'Systematic Review']['year'].unique()

prev = uniquePeriods[0]
for i, y in enumerate(uniquePeriods):
    nodes = nodesdf[nodesdf['year'] <= y]
    edges = edgesdf[edgesdf['source'].isin(nodes['id'])]
    #nodes['outline'] = np.where(nodes.year < y, 'black', 'red')

    SRperiods.append({'endyear': y, 'nodes': nodes, 'edges': edges})


# drawing with nx.draw_networkx_{nodes|edges}
# this way requires that the subsets be dictionaries where the
# keys are the 'id' and the values are the coordinate pairs 
for i, period in enumerate(SRperiods):
    # TODO annoyingly nx.draw turns off padding, margins differently
   
    # SRs are split from PSRs because unlike coloring, shape can only
    # be a single option per draw function.
    SRs = period['nodes'][period['nodes']['Type'] == 'Systematic Review']
    PSRs = period['nodes'][period['nodes']['Type'] == 'Primary Study Report']
    SRnodepos = dict(SRs[['id', 'coords']].values)
    PSRnodepos = dict(PSRs[['id', 'coords']].values)

    # want to do an edgecolor on nodes that are new, whether SR or PSR,
    # and no edgecolor on older nodes
    # split edges based on if they're new this period, do this with
    # a join on the previous period, i.e., old things are present 
    # in the previous period. pandas doesn't have a direct anti-join, sigh...
    # can't check on year because edges don't have years
    nodepos = dict(period['nodes'][['id', 'coords']].values)
    if i > 0:
        current_nodes= period['nodes']
        previous_nodes= SRperiods[i-1]['nodes']
        tmp = current_nodes.merge(previous_nodes, how='outer', indicator=True)
        new_nodes= tmp[tmp['_merge'] == 'left_only']
        old_nodes= tmp[tmp['_merge'] == 'both']

        new_SRs = new_nodes[new_nodes['Type'] == 'Systematic Review']
        old_SRs = old_nodes[old_nodes['Type'] == 'Systematic Review']

        new_PSRs = new_nodes[new_nodes['Type'] == 'Primary Study Report']
        old_PSRs = old_nodes[old_nodes['Type'] == 'Primary Study Report']
    
        new_SRs_pos = dict(new_SRs[['id', 'coords']].values)
        old_SRs_pos = dict(old_SRs[['id', 'coords']].values)

        new_PSRs_pos = dict(new_PSRs[['id', 'coords']].values)
        old_PSRs_pos = dict(old_PSRs[['id', 'coords']].values)

        nx.draw_networkx_nodes(G, new_SRs_pos, nodelist=new_SRs_pos.keys(),
            node_color=new_SRs['fill'].to_list(), node_shape='s', edgecolors='red')
        nx.draw_networkx_nodes(G, old_SRs_pos, nodelist=old_SRs_pos.keys(),
            node_color=old_SRs['fill'].to_list(), node_shape='s')
        
        nx.draw_networkx_nodes(G, new_PSRs_pos, nodelist=new_PSRs_pos.keys(),
            node_color=new_PSRs['fill'].to_list(), edgecolors='red')
        nx.draw_networkx_nodes(G, old_PSRs_pos, nodelist=old_PSRs_pos.keys(),
            node_color=old_PSRs['fill'].to_list())
        

        current_edges = period['edges']
        previous_edges = SRperiods[i-1]['edges']
        tmp = current_edges.merge(previous_edges, how='outer', indicator=True)
        new_edges = tmp[tmp['_merge'] == 'left_only']
        old_edges = tmp[tmp['_merge'] == 'both']
        
        nx.draw_networkx_edges(G, nodepos, edgelist=old_edges['tuples'].to_list(), edge_color='darkgray')
        nx.draw_networkx_edges(G, nodepos, edgelist=new_edges['tuples'].to_list(), edge_color='red')
    else:
        nx.draw_networkx_nodes(G, SRnodepos, nodelist=SRnodepos.keys(),
                node_color=SRs['fill'].to_list(), node_shape='s')
        nx.draw_networkx_nodes(G, PSRnodepos, nodelist=PSRnodepos.keys(),
                node_color=PSRs['fill'].to_list())
        nx.draw_networkx_edges(G, nodepos, period['edges']['tuples'].to_list(), edge_color='darkgray')

    # labels are all drawn with the same style
    # so no separation is necessary
    nx.draw_networkx_labels(G, nodepos, labels = dict(period['nodes'][['id', 'labels']].values))
    

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('test-{}.png'.format(i), pad_inches=0, bbox_inches='tight')
    
    plt.close()
