#!/usr/bin/env python3

# Example application drawing an inclusion network. Options, paths, etc.
# are set in a YAML config file.
import InclusionNetwork
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''draw an inclusion network 
    evolution over time''')
    parser.add_argument('cfgyaml', help='path to YAML config file')
    args = parser.parse_args()

    network = InclusionNetwork.InclusionNetwork()
    network.load_cfgs(args.cfgyaml)
    network.load_nodes()
    network.load_edges()
    network.draw()
