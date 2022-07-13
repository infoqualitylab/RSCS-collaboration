# Example application cycling through all the available pygraphviz layouts
# which will work (but not necessarily make sense) for the ExRx or Salt 
# Controversy data.
import CoauthorNetwork
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw a coauthor network')
    parser.add_argument('cfgyaml', help='path to YAML config file')
    args = parser.parse_args()

    # coauthur network hangs on dot and circo layouts.
    layouts = ['neato', 'twopi', 'fdp', 'sfdp']
    for layout in layouts:
        print(f'starting on {layout} layout')
        network = CoauthorNetwork.CoauthorNetwork(engine=layout)
        network.load_cfgs(args.cfgyaml)
        network.load_nodes()
        network.load_edges()
        network.create_graph()
        network.layout_graph()
        network.set_node_aesthetics()
        network.set_edge_aesthetics()
        network.draw()
