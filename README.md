# IQL/RSCS collaboration on visualizing inclusion networks and coauthor networks 


The are three class files: 
1. IQLNetwork.py - defines methods for loading data, creating networkX Graph object, etc.
2. InclusionNetwork.py - subclasses IQLNetwork to draw an implicitly dynamic graph in a tiled layout
3. CoauthorNetwor.py - subclasses IQLNetwork to draw either an entire coauthor network or the two largest connected components.

There are three scripts showing how these classes can be used:
1. inclusion_pygraphviz_layout.py - draws inclusion network using networkX+pygraphviz layouts
2. inclusion_precomputed_layout.py - draws inclusion network using a JSON layout exported from Gephi
3. coauth_layouts.py - draws coauthor network using networkX

Each script also expects a YAML config file for setting various things like paths to data, etc. The purpose of these is to avoid hand-editing the class files for common changes.

run as, e.g.:

$ python3 coauth_layouts.py exrx-coauth.yml

Output PNGs into current directory.
