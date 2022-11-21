# Information Quality Lab/Research Software Collaborative Services collaboration on visualizing inclusion networks and coauthor networks 


The are three class files: 
1. IQLNetwork.py - defines methods for loading data, creating networkX Graph object, etc.
2. InclusionNetwork.py - subclasses IQLNetwork to draw an implicitly dynamic graph in a tiled layout or as a sequence of images.
3. CoauthorNetwork.py (EXPERIMENTAL) - subclasses IQLNetwork to draw either an entire coauthor network or the two largest connected components.

There are three scripts showing how these classes can be used:
1. draw_inclusion_network.py 
2. draw_coauthor_network.py 
2. draw_inclusion_precomputed_layout.py - draws inclusion network using a JSON layout exported from Gephi

Each script also expects a YAML config file for setting various things like paths to data, etc. The purpose of these is to avoid hand-editing the class files for common changes.

run as, e.g.:

$ python3 draw_inclusion_network.py exrx-2022-09-08.yml

Outputs PNG(s) into current directory.

## YAML configuration 
The YAML file given on the command line is read and all YAML attributes are converted to Python class attributes using setattr. Strings don't have to necessarily be quoted, but must be if they contain white space, e.g., in file paths.

Configs in the YAML which are not explicitly handled by the Python code are ignored.

Configs are generally of three types:
1. input meta data
2. graphical meta data
3. graphical options

In practice, the idea is that if there's going to be any cutting and pasting going on, it's of config files and not of the Python itself.
