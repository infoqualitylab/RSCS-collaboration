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
