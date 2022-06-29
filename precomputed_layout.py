import InclusionNetwork
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw an inclusion network evolution over time')
    parser.add_argument('cfgyaml', help='path to YAML config file')
    args = parser.parse_args()

    network = InclusionNetwork.InclusionNetwork(engine='precomputed')
    network.load_cfgs(args.cfgyaml)
    network.load_nodes()
    network.load_edges()
    network.create_graph()
    network.load_layout_json()
    network.set_aesthetics()
    network.draw_graph_evolution()
