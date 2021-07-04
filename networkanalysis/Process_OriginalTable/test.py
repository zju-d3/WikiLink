from Private import PubFunctions
import networkx as nx
import time
import main


def test_ReadNeighborsFromMysql():
    cnx, rcursor = PubFunctions.creatCursor('abcdeijm_test', 'R')

    sG1 = nx.Graph()
    t1s = time.time()
    PubFunctions.DFS_recursive_LevelNeighbor(500, sG1, rcursor, 'all_w2w', maxlevel=1)
    t1e = time.time()

    sG2 = nx.Graph()
    t2s = time.time()
    PubFunctions.BFS_LevelNeighbor(500, sG2, rcursor, 'all_w2w', maxlevel=1)
    t2e = time.time()

    sG3 = nx.Graph()
    t3s = time.time()
    PubFunctions.Complex_DFS_LevelNeighbor(500, sG3, rcursor, 'all_w2w', 1)
    t3e = time.time()

    print(len(sG1.edges()), len(sG1.nodes()), t1e - t1s)

    print(len(sG2.edges()), len(sG2.nodes()), t2e - t2s)

    print(len(sG3.edges()), len(sG3.nodes()), t3e - t3s)

    return


def test_allweight():
    G = nx.Graph()
    G.add_edges_from(
        [(1, 2, {'weight': 1.0}), (1, 3, {'weight': 3.0}), (3, 2, {'weight': 2.0}), (3, 4, {'weight': 4.0})])

    return


def rawGraph_withAlpha():
    """
    load raw data and add disparity alpha value
    :return:
    """
    schema = 'total_v3_csvneo4j'
    reltable = 'all_w2w'
    labtable = 'all_keywords'
    Graph_type = 'undirected'

    G = main.load_rawGraph(schema, reltable, labtable, Graph_type=Graph_type)
    G = main.disparity_alpha(G)
    print('finish add alpha')
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()), '\n')

    nx.write_gpickle(G, '../RawGraphWithDisparityAlpha_{}_{}.gpickle'.format(Graph_type, schema))
    return


def filter_edgeandNode(alpha_thred, nodeDegree_thred):
    """
    filter edge based on disparity alpha.
    filter node based on node degree
    :param alpha_thred: threshold of alpha
    :param nodeDegree_thred: threshold of node degree
    :return:
    """
    schema = 'total_v3_csvneo4j'
    Graph_type = 'undirected'

    G = nx.read_gpickle('../RawGraphWithDisparityAlpha_{}_{}.gpickle'.format(Graph_type, schema))
    G = main.disparity_filter(G, alpha_thred)
    G = main.nodeDegree_filter(G, nodeDegree_thred)
    nx.write_gpickle(G, '../filteredG_{}_alpha{}_nodeD{}_{}.gpickle'.format(Graph_type, alpha_thred, nodeDegree_thred,
                                                                            schema))
    print('-----', alpha_thred, '----------')
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()))

    return


def addNodeDe_EdgeDist():
    """
    Add node degree and edge distance on the filtered Graph
    :return: Graph
    """
    schema = 'total_v3_csvneo4j'
    Graph_type = 'undirected'
    alpha_thred = 0.65
    nodeDegree_thred = 1.0
    DisTypes = ['R']
    NormalizaMethod = ['r', 'n']
    mean_methods = ['GM', 'HM']

    G = nx.read_gpickle(
        '../filteredG_{}_alpha{}_nodeD{}_{}.gpickle'.format(Graph_type, alpha_thred, nodeDegree_thred, schema))
    print('after read')
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()))
    G = main.addNode_degree(G)
    print('finish adding node degree')
    G = main.addEdge_distance(G, DisTypes, NormalizaMethod, mean_methods)
    print('finish adding edge degree')

    nx.write_gpickle(G, '../addNodeEdgeDegree_{}_{}_alpha{}_nodeD{}_{}.gpickle'.format(
        '+'.join([''.join(DisTypes), ''.join(NormalizaMethod), ''.join(mean_methods)]), Graph_type, alpha_thred,
        nodeDegree_thred, schema))
    print('finishing write gpickle')
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()))

    return
