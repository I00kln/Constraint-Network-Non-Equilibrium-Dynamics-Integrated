#ifndef GRAPH_TOPOLOGY_H
#define GRAPH_TOPOLOGY_H

#include "constraint_network.h"

void init_tetrahedron_graph(Graph *graph);
void init_octahedron_graph(Graph *graph);
void init_cube_graph(Graph *graph);
void print_graph_info(const Graph *graph);
int find_edge(const Graph *graph, int i, int j);
void get_edge_direction(const Graph *graph, int edge_idx, int *from, int *to);

#endif
