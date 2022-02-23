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

# add labels
edgesdf = pd.merge(edgesdf, nodecoords, left_on='source', right_on='id')
edgesdf = edgesdf.drop(['id'],axis=1)
edgesdf = pd.merge(edgesdf, nodecoords, 
        left_on='target', right_on='id', 
        suffixes=tuple(['_source', '_target']))
edgesdf = edgesdf.drop(['id'], axis=1)

# tuples for edgelist
edgesdf['tuples'] = tuple(zip(edgesdf.source, edgesdf.target))

# get the unique SR years to make cohorts
SRs = nodesdf[nodesdf['Type'] == 'Systematic Review']

# loop over unique SR years grabbing just nodes <= y
SRperiods = []
for y in SRs['year'].unique():
    nodes = nodesdf[nodesdf['year'] <= y]

    edges = edgesdf[edgesdf['source'].isin(nodes['id'])]
    SRperiods.append({'nodes': nodes, 'edges': edges})


# drawing directly with matplotlib
# issues with this approach
# have to do too much work on things like scaling the arrows,
# nodes, etc. 
'''
i = 0
for period in SRperiods:

    fig, ax = plt.subplots(figsize=(1280/96, 720/96), dpi=96)

    # draw edges using patches
    for idx, row in period['edges'].iterrows():
        arrow = mpatches.FancyArrowPatch(row['source_coords'],
                                         row['target_coords'],
                                         mutation_scale=10,
                                         arrowstyle='-|>',
                                         color='gainsboro',
                                         zorder=1)
        ax.add_patch(arrow)

    # draw the nodes
    # color by attitude
    plt.scatter(period['nodes']['x'], period['nodes']['y'],
            s=150.0,
            c=period['nodes']['fill'], zorder=2)

    # draw the labels, just id for now
    for idx, row in period['nodes'].iterrows():
        plt.annotate(row['id'], (row['x'], row['y']), color='gray')
    plt.axis('off')
    plt.savefig('backwards-scatter-{}.png'.format(i))
    plt.close()
    i += 1
'''

# drawing with nx.draw_networkx_{nodes|edges}
# this way requires that the subsets be dictionaries where the
# keys are the 'id' and the values are the coordinate pairs 
i = 0
for period in SRperiods:
    # this is ugly, if it works figure out a better storage
    # we get back to the issue here that
    # G.nodes is 1:14, 26:93
    # so pd.DataFrame.to_dict() is not going to do what we want,
    # however we can throw it into the dict ctor and be fine
    nodepos = dict(period['nodes'][['id', 'coords']].values)

    # annoyingly nx.draw turns off padding, margins differently
    nx.draw_networkx_nodes(G, nodepos, nodelist=nodepos.keys())
    nx.draw_networkx_edges(G, nodepos, edgelist=period['edges']['tuples'].to_list())

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('test-{}.png'.format(i), pad_inches=0, bbox_inches='tight')
    
    plt.close()
    i+=1
