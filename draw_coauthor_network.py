#!/usr/bin/env python3

# Example application for drawing connected components of a coauthor network
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
    network.draw(useCmap='')
