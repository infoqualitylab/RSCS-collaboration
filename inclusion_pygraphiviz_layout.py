# Example application cycling through all the available pygraphviz layouts
# which will work (but not necessarily make sense) for the ExRx or Salt 
# Controversy data.
import InclusionNetwork
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw an inclusion network evolution over time')
    parser.add_argument('cfgyaml', help='path to YAML config file')
    args = parser.parse_args()

    network = InclusionNetwork.InclusionNetwork()
    network.load_cfgs(args.cfgyaml)
    network.load_nodes()
    network.load_edges()
    network.set_aesthetics()
    network.draw()
