#include "graph_topology.h"

void init_tetrahedron_graph(Graph *graph) {
    graph->num_nodes = 4;
    graph->num_edges = 6;
    graph->num_faces = 4;

    graph->edges[0] = (Edge){0, 1};
    graph->edges[1] = (Edge){0, 2};
    graph->edges[2] = (Edge){0, 3};
    graph->edges[3] = (Edge){1, 2};
    graph->edges[4] = (Edge){1, 3};
    graph->edges[5] = (Edge){2, 3};

    graph->faces[0] = (Face){{0, 3, 1}, {0, 1, 2}};
    graph->faces[1] = (Face){{0, 4, 2}, {0, 1, 3}};
    graph->faces[2] = (Face){{1, 5, 3}, {0, 2, 3}};
    graph->faces[3] = (Face){{3, 5, 4}, {1, 2, 3}};

    for (int i = 0; i < graph->num_nodes; i++) {
        graph->node_edge_count[i] = 0;
    }

    for (int e = 0; e < graph->num_edges; e++) {
        int from = graph->edges[e].i;
        int to = graph->edges[e].j;
        graph->node_edges[from][graph->node_edge_count[from]++] = e;
        graph->node_edges[to][graph->node_edge_count[to]++] = e;
    }
}

void init_octahedron_graph(Graph *graph) {
    graph->num_nodes = 6;
    graph->num_edges = 12;
    graph->num_faces = 8;

    graph->edges[0] = (Edge){0, 1};
    graph->edges[1] = (Edge){0, 2};
    graph->edges[2] = (Edge){0, 3};
    graph->edges[3] = (Edge){0, 4};
    graph->edges[4] = (Edge){1, 2};
    graph->edges[5] = (Edge){2, 3};
    graph->edges[6] = (Edge){3, 4};
    graph->edges[7] = (Edge){4, 1};
    graph->edges[8] = (Edge){5, 1};
    graph->edges[9] = (Edge){5, 2};
    graph->edges[10] = (Edge){5, 3};
    graph->edges[11] = (Edge){5, 4};

    graph->faces[0] = (Face){{0, 1, 4}, {0, 1, 2}};
    graph->faces[1] = (Face){{1, 2, 5}, {0, 2, 3}};
    graph->faces[2] = (Face){{2, 3, 6}, {0, 3, 4}};
    graph->faces[3] = (Face){{3, 0, 7}, {0, 4, 1}};
    graph->faces[4] = (Face){{8, 9, 4}, {5, 2, 1}};
    graph->faces[5] = (Face){{9, 10, 5}, {5, 3, 2}};
    graph->faces[6] = (Face){{10, 11, 6}, {5, 4, 3}};
    graph->faces[7] = (Face){{11, 8, 7}, {5, 1, 4}};

    for (int i = 0; i < graph->num_nodes; i++) {
        graph->node_edge_count[i] = 0;
    }

    for (int e = 0; e < graph->num_edges; e++) {
        int from = graph->edges[e].i;
        int to = graph->edges[e].j;
        graph->node_edges[from][graph->node_edge_count[from]++] = e;
        graph->node_edges[to][graph->node_edge_count[to]++] = e;
    }
}

void init_cube_graph(Graph *graph) {
    graph->num_nodes = 8;
    graph->num_edges = 18;
    graph->num_faces = 12;

    graph->edges[0] = (Edge){0, 1};
    graph->edges[1] = (Edge){1, 2};
    graph->edges[2] = (Edge){2, 3};
    graph->edges[3] = (Edge){3, 0};
    graph->edges[4] = (Edge){4, 5};
    graph->edges[5] = (Edge){5, 6};
    graph->edges[6] = (Edge){6, 7};
    graph->edges[7] = (Edge){7, 4};
    graph->edges[8] = (Edge){0, 4};
    graph->edges[9] = (Edge){1, 5};
    graph->edges[10] = (Edge){2, 6};
    graph->edges[11] = (Edge){3, 7};
    graph->edges[12] = (Edge){0, 5};
    graph->edges[13] = (Edge){1, 6};
    graph->edges[14] = (Edge){2, 7};
    graph->edges[15] = (Edge){3, 4};
    graph->edges[16] = (Edge){0, 2};
    graph->edges[17] = (Edge){4, 6};

    graph->faces[0] = (Face){{0, 1, 12}, {0, 1, 5}};
    graph->faces[1] = (Face){{12, 4, 8}, {0, 5, 4}};
    graph->faces[2] = (Face){{1, 2, 13}, {1, 2, 6}};
    graph->faces[3] = (Face){{13, 5, 9}, {1, 6, 5}};
    graph->faces[4] = (Face){{2, 3, 14}, {2, 3, 7}};
    graph->faces[5] = (Face){{14, 6, 10}, {2, 7, 6}};
    graph->faces[6] = (Face){{3, 0, 15}, {3, 0, 4}};
    graph->faces[7] = (Face){{15, 7, 11}, {3, 4, 7}};
    graph->faces[8] = (Face){{0, 2, 16}, {0, 2, 3}};
    graph->faces[9] = (Face){{0, 16, 3}, {0, 3, 2}};
    graph->faces[10] = (Face){{4, 5, 17}, {4, 5, 6}};
    graph->faces[11] = (Face){{4, 17, 6}, {4, 6, 7}};

    for (int i = 0; i < graph->num_nodes; i++) {
        graph->node_edge_count[i] = 0;
    }

    for (int e = 0; e < graph->num_edges; e++) {
        int from = graph->edges[e].i;
        int to = graph->edges[e].j;
        graph->node_edges[from][graph->node_edge_count[from]++] = e;
        graph->node_edges[to][graph->node_edge_count[to]++] = e;
    }
}

void print_graph_info(const Graph *graph) {
    printf("=== Graph Topology ===\n");
    printf("Nodes: %d, Edges: %d, Faces: %d\n\n", 
           graph->num_nodes, graph->num_edges, graph->num_faces);
    
    printf("Edges:\n");
    for (int e = 0; e < graph->num_edges; e++) {
        printf("  Edge %d: %d -> %d\n", e, 
               graph->edges[e].i, graph->edges[e].j);
    }
    
    printf("\nFaces:\n");
    for (int f = 0; f < graph->num_faces; f++) {
        printf("  Face %d: edges [%d, %d, %d], nodes [%d, %d, %d]\n",
               f, graph->faces[f].edges[0], graph->faces[f].edges[1],
               graph->faces[f].edges[2], graph->faces[f].nodes[0],
               graph->faces[f].nodes[1], graph->faces[f].nodes[2]);
    }
    
    printf("\nNode connectivity:\n");
    for (int n = 0; n < graph->num_nodes; n++) {
        printf("  Node %d: %d edges [", n, graph->node_edge_count[n]);
        for (int i = 0; i < graph->node_edge_count[n]; i++) {
            printf("%d", graph->node_edges[n][i]);
            if (i < graph->node_edge_count[n] - 1) printf(", ");
        }
        printf("]\n");
    }
}

int find_edge(const Graph *graph, int i, int j) {
    for (int e = 0; e < graph->num_edges; e++) {
        if ((graph->edges[e].i == i && graph->edges[e].j == j) ||
            (graph->edges[e].i == j && graph->edges[e].j == i)) {
            return e;
        }
    }
    return -1;
}

void get_edge_direction(const Graph *graph, int edge_idx, int *from, int *to) {
    *from = graph->edges[edge_idx].i;
    *to = graph->edges[edge_idx].j;
}
