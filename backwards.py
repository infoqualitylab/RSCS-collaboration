# plan:
# - read data into node/edges dataframe
# - use a graph algo to layout final, complete inclusion network
# - grab node coords from this layout
# - draw graph evolution directly
#   - loop over years of Study Reviews
#   - draw relevant subset of nodes/edges using previously
#     generated coords

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

# load data
datapath = '~/Vis/projects/schneider/data/derived/'
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
#fullgraphpos = nx.spring_layout(G)
fullgraphpos = nx.nx_agraph.graphviz_layout(G)
nx.draw(G, fullgraphpos)
plt.savefig('backwards-fullgraph.png')
plt.close()

# get the unique SR years to make cohorts
SRs = nodesdf[nodesdf['Type'] == 'Systematic Review']
SRyears = SRs['year'].unique()

# loop over SRyears grabbing just nodes <= y
cohorts = []
for y in SRyears:
    nodes = nodesdf[nodesdf['year'] <= y]
    edges = edgesdf[edgesdf['source'].isin(nodes['id'])]
    cohorts.append(tuple([nodes, edges]))

# now I should be able to use each cohort to make a subgraph view?
# or grab that subset from G somehow...
# fullgraphpos is a dict. of nodeid:coords

# feel like there should be a more elegant way to do this...
i = 0

draw_attrs = { 'with_labels': True,
        'node_size': 150,
        'width': 0.5,
        'font_size': 9,
        'alpha': 0.5
        }
for cohort in cohorts:
    subG = nx.DiGraph()
    cohortpos = { x: fullgraphpos[x] for x in cohort[0]['id'] }

    for idx,row in cohort[0].iterrows():
        subG.add_node(row['id'], **row)
    srcs = cohort[1]['source'].tolist()
    targs = cohort[1]['target'].tolist()
    subG.add_edges_from(zip(srcs,targs))

    nx.draw(subG, pos=cohortpos, **draw_attrs)
    plt.savefig('backwards-{}.png'.format(i))
    plt.close()
    i += 1
    del(cohortpos)
    del(subG)


'''
# what if I draw directly in matplotlib, e.g. as a scatter plot?
# this works, but then I have to explicitly draw edges and turn off
# all the annotations, etc. but it might be open using a pygraphviz
# layout algo that's better than nx.spring_layout()
i = 0
for cohort in cohorts:
    cohortpos = {x : fullgraphpos[x] for x in cohort[0]['id']}
    cohortx = deepcopy([x for x,y in cohortpos.values()])
    cohorty = deepcopy([y for x,y in cohortpos.values()])
    print('{} {}'.format(len(cohortx), len(cohorty)))
    print(cohortx)
    plt.scatter(cohortx, cohorty)
    plt.savefig('backwards-scatter-{}.png'.format(i))
    i += 1
'''
