# RSCS collaboration on visualizing inclusion networks

run as:

$ python3 inclusion-net-vis.py &lt;config&gt;

where config is a YAML file specifying the mapping of data column names and
listing paths to the data. Compare the two YAML files in the repo for details.

Output is a PNG of tiled subplots in the current directory.

Tested using Article_attr.2.csv from 2021-09-08 and inclusion_net_edges.csv from 2021-05-29,
and the Exercise and Depression data from March 2022.
