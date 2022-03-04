# RSCS collaboration on visualizing inclusion networks

run as:

$ python3 inclusion-net-vis.py <nodescsv> <edgescsv>

where nodescsv at least has columns: 

ID,year,Attitude,Type

and edgescsv at least has columns:

citing_ID,cited_ID

which are used as the source and target in a directed graph, respectively.

Output is currently a series of PNG files in the current directory.

Tested using Article_attr.2.csv from 2021-09-08 and inclusion_net_edges.csv from 2021-05-29.


