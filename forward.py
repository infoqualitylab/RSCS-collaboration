# plan:
# - read data into node/edges dataframe
# - make data frames for each SR year "cohort"
# - draw graph evolution directly
#   - loop over cohort years of Study Reviews
#   - draw relevant subset of nodes/edges using previously
#     generated coords


# This works, after a fashsion, I can see the evolution of the
# network and things stay more or less in place (though the
# adjustments to fit the window can be weird) however there
# is an additional factor that is trying to "balance", unsure,
# disconnected components and that is not looking great.

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
# just noticing that some of the layout algos take pos and fixed args
# so maybe I can do this going forwards and just need to be able
# to group by SR dates...

SRs = nodesdf[nodesdf['Type'] == 'Systematic Review']
SRyears = SRs['year'].unique()

# loop over SRyears grabbing just nodes <= y
cohorts = []

# this is just nodes, really also want to grab edges for that
# cohort, too. Edges are only from SRs to PSRs, so I should be 
# able to grab just the 
for y in SRyears:
    nodes = nodesdf[nodesdf['year'] <= y]
    edges = edgesdf[edgesdf['source'].isin(nodes['id'])]
    cohorts.append(tuple([nodes, edges]))

# loop over cohorts, using coords from previous iteration as
# fixed points in next

prevpos = None
fixed = None
init = True

i = 0
for cohort in cohorts:

    subG = nx.DiGraph()
    for idx, row in cohort[0].iterrows():
        subG.add_node(row['id'], **row)
    srcs = cohort[1]['source'].tolist()
    targs = cohort[1]['target'].tolist()
    subG.add_edges_from(zip(srcs, targs))

    if init:
        prevpos = nx.spring_layout(subG,seed=42)
        prevfixed = deepcopy(subG.nodes)
        init = False
    else:
        prevpos = nx.spring_layout(subG,pos=prevpos,fixed=prevfixed,seed=42)
        prevfixed = deepcopy(subG.nodes)

    nx.draw(subG, pos=prevpos)
    plt.savefig('forwards-{}.png'.format(i))
    i += 1
    
