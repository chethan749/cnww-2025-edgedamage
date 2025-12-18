# -*- coding: utf-8 -*-
# =============================================================================
# Import libraries

import math as m
import osmnx as ox
import argparse
ox.__version__

# =============================================================================
# create the network

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='beirut', help='name of the city')
parser.add_argument('--lat', type=float, default=33.8938, help='latitude of the city center')
parser.add_argument('--lon', type=float, default=35.5018, help='longitude of the city center')
parser.add_argument('--area', type=float, default=10000, help='area (in km^2) of the bounding box to create the street network')
args = parser.parse_args()

name = args.name
point = (args.lat, args.lon)
distance = m.sqrt(args.area * 1000 * 1000)/2

#G = ox.graph.graph_from_place('paris', network_type='drive')
G = ox.graph.graph_from_point(point, distance, dist_type='bbox', network_type='drive') # multidigraph
lcc = ox.utils_graph.get_largest_component(G)
ox.plot.plot_graph(lcc, node_size=0)

ox.io.save_graphml(lcc, filepath='../output/'+name+'.graphml')