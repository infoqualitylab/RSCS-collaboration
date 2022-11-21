# Example application cycling through all the available pygraphviz layouts
# which will work (but not necessarily make sense) for the ExRx or Salt 
# Controversy data.
import CoauthorNetwork
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw a coauthor network')
    parser.add_argument('cfgyaml', help='path to YAML config file')
    args = parser.parse_args()

    network = CoauthorNetwork.CoauthorNetwork()
    network.load_cfgs(args.cfgyaml)
    network.load_nodes()
    network.load_edges()
    network.create_graph()
    network.filter_connected_components()
    network.layout_graph()
    network.draw(useCmap='')
