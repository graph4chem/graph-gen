// CPP program to find degree of a vertex.
#include <iostream>

using namespace std;

// structure of a graph
struct graph
{
    // vertices
    int v;
    // edges
    int e;
    // direction from src to des
    int **dir;
};

// Returns degree of ver in given graph
int findDegree(struct graph *G, int ver)
{
    // Traverse through row of ver and count
    // all connected cells (with value 1)
    int degree = 0;
    for (int i = 0; i < G->v; i++)

        // if src to des is 1 the degree count
        if (G->dir[ver][i] == 1)
            degree++;

    // below line is to account for self loop in graph
    // check sum of degrees in graph theorem
    if (G->dir[ver][ver] == 1)
        degree++;
    return degree;
}

struct graph *createGraph(int v, int e)
{
    // G is a pointer of a graph
    struct graph *G = new graph;

    G->v = v;
    G->e = e;

    // allocate memory
    G->dir = new int *[v];

    for (int i = 0; i < v; i++)
        G->dir[i] = new int[v];

    /*  0-----1
        | \   |
        |  \  |
        |   \ |
        2-----3     */

    //direction from 0
    G->dir[0][1] = 1;
    G->dir[0][2] = 1;
    G->dir[0][3] = 1;

    //direction from 1
    G->dir[1][0] = 1;
    G->dir[1][3] = 1;

    //direction from 2
    G->dir[2][0] = 1;
    G->dir[2][3] = 1;

    //direction from 3
    G->dir[3][0] = 1;
    G->dir[3][1] = 1;
    G->dir[3][2] = 1;

    return G;
}

// Driver code
int main()
{
    int vertices = 4;
    int edges = 5;
    struct graph *G = createGraph(vertices, edges);

    // loc is find the degree of
    // particular vertex
    int ver = 0;

    int degree = findDegree(G, ver);
    cout << degree << "\n";
    return 0;
}