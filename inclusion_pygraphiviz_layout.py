# Example application cycling through all the available pygraphviz layouts
# which will work (but not necessarily make sense) for the ExRx or Salt 
# Controversy data.
import InclusionNetwork
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw an inclusion network evolution over time')
    parser.add_argument('cfgyaml', help='path to YAML config file')
    args = parser.parse_args()

    layouts = ['neato', 'dot', 'twopi', 'circo', 'fdp', 'sfdp']
    for layout in layouts:
        network = InclusionNetwork.InclusionNetwork(engine=layout)
        network.load_cfgs(args.cfgyaml)
        network.load_nodes()
        network.load_edges()
        network.create_graph()
        network.layout_graph()
        network.set_aesthetics()
        network.draw_graph_evolution()
