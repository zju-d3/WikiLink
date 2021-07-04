from Private.PubFunctions import unweight_allocation
from Private import PubFunctions, PF
import networkx as nx
import datetime
import math
import time
from math import fabs, log, sqrt


def write_undirected(schema, reltable, labtable):
    uG = PubFunctions.loadw2wdict(schema, reltable, 'undirected')
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': undirected graph readed')

    uG = PubFunctions.uG_to_uGuW(uG, 'undirected')
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': undirected graph add weights')

    uG = PubFunctions.load_nodelabel(uG, schema, labtable)
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': undirected graph add node label')

    nx.write_gpickle(uG, 'undirected.gpickle')
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': undirected graph written')

    return




def write_onedirected(schema, reltable, labtable, tp):
    dG = PubFunctions.loadw2wdict(schema, reltable, 'one-directed')
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': one-directed graph readed')

    dG = PubFunctions.dG_to_dGuW(dG, tp, 'one-directed')
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': one-directed graph add weights')

    dG = PubFunctions.load_nodelabel(dG, schema, labtable)
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'),
          ': one-directed graph add node label')

    nx.write_gpickle(dG, 'onedirected_{}.gpickle'.format(tp))
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': one-directed graph written')

    return


def write_bidirected(schema, reltable, labtable, tp):
    bdG = PubFunctions.loadw2wdict(schema, reltable, 'bi-directed')
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': bi-directed graph readed')

    bdG = PubFunctions.bdG_to_bdGdW(bdG, tp, 'bi-directed')
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': bi-directed graph add weights')

    bdG = PubFunctions.load_nodelabel(bdG, schema, labtable)
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'),
          ': bi-directed graph add node label')

    nx.write_gpickle(bdG, 'bidirected_{}.gpickle'.format(tp))
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': bi-directed graph written')

    return


# The main function use original keywords table and relations table of a schema.
# And create three forms of graph:
# 1. undirected graph with undirected weights and domian dissimilarity
# 2. one-directed graph with undirected weights, domian dissimilarity, and G2S direction
# 3. bi-directed graph with forwards and backwards directed weights
# Finally, write these three graphs into gpickle
# Checked OK
def main(schema, reltable, labtable, tp):
    write_undirected(schema, reltable, labtable)
    write_onedirected(schema, reltable, labtable, tp)
    write_bidirected(schema, reltable, labtable, tp)

    print('All finished')

    return


def reduceGraph(read_g, write_g, minEdgeWeight, minNodeDegree, Lp, Sp):
    """
    Simplify the undirected graph and then update the 3 undirected weight properties.
    :param read_g: is the graph pickle to read
    :param write_g: is the updated graph pickle to write
    :param minEdgeWeight: the original weight of each edge should be >= minEdgeWeight
    :param minNodeDegree: the degree of each node should be >= minNodeDegree. the degree here is G.degree(node), NOT G.degree(node,weight='weight)
    :return: None
    """
    G = nx.read_gpickle(read_g)
    print('number of original nodes: ', nx.number_of_nodes(G))
    print('number of original edges: ', nx.number_of_edges(G))

    for (u, v, w) in G.edges(data='weight'):
        if w < minEdgeWeight:
            G.remove_edge(u, v)

    for n in G.nodes():
        if G.degree(n) < minNodeDegree:
            G.remove_node(n)

    print('number of new nodes: ', nx.number_of_nodes(G))
    print('number of new edges: ', nx.number_of_edges(G))

    for (a, b, w) in G.edges_iter(data='weight'):
        unweight_allocation(G, a, b, w, Lp, Sp)

    print('update weight ok')
    nx.write_gpickle(G, write_g)

    return


def load_rawGraph(schema, reltable, labtable, Graph_type='undirected'):
    """
    load raw graph from a schema
    :param schema: the mysql schema storing the data
    :param reltable: the name of edge table
    :param labtable: the name of keywords lable table
    :param Graph_type: the graph type: 'undirected' or 'one-directed'
    :return: networkx Graph
    """
    # load graph by edge table
    G = PubFunctions.loadw2wdict(schema, reltable, Graph_type)
    print('finish loading raw edges', time.strftime('%Y-%m-%d %H:%M:%S'))
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()), '\n')
    # add node label
    G = PubFunctions.load_nodelabel(G, schema, labtable)
    print('add node label', time.strftime('%Y-%m-%d %H:%M:%S'))
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()), '\n')

    return G


def disparity_alpha(G):
    """
    calculate the disparity maxalpha and minalpha value for each edges
    :param G: networkx Graph
    :return: G
    """

    def get_maxAlpha(G, i, j, w):
        Si = G.degree(i, weight='weight')
        Pij = w / Si
        Ki = float(G.degree(i))
        if Ki > 1:
            alpha_i = 1.0 - math.pow((1.0 - Pij), (Ki - 1))
        else:
            # if let alpha be 0, remove edge of nodes whose degree<=1
            # if let alpha be 1, keep edge of nodes whose degree<=1
            alpha_i = 0.0

        Sj = G.degree(j, weight='weight')
        Pji = w / Sj
        Kj = float(G.degree(j))
        if Kj > 1:
            alpha_j = 1.0 - math.pow((1.0 - Pji), (Kj - 1))
        else:
            # if let alpha be 0, remove edge of nodes whose degree<=1
            # if let alpha be 1, keep edge of nodes whose degree<=1
            alpha_j = 0.0

        return max(alpha_i, alpha_j), min(alpha_i, alpha_j)

    # calculate alpha
    for (a, b, w) in G.edges(data='weight'):
        maxalpha, minalpha = get_maxAlpha(G, a, b, w)
        G[a][b]['maxAlpha'] = maxalpha
        G[a][b]['minAlpha'] = minalpha
    print('finish calculate alpha', time.strftime('%Y-%m-%d %H:%M:%S'))
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()), '\n')

    return G


def disparity_filter(G, alpha_thred):
    """
    Network reduction by disparity filter.
    :param G: networkx graph to be reduced
    :param alpha_thred: the edges with maxAlpha >= alpha_thred will be preserved
    :return: G
    """
    # disparity_filter
    for (a, b) in G.edges():
        if G[a][b]['maxAlpha'] < alpha_thred:
            G.remove_edge(a, b)
    print('finish filter', time.strftime('%Y-%m-%d %H:%M:%S'))
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()), '\n')

    return G


def globalEdge_filter(G, minorgEdgeWeight):
    """
    filter the graph based on minimum original weight
    :param G: networkx graph
    :param minorgEdgeWeight: minimum original weight
    :return: G
    """
    for (u, v, w) in G.edges(data='weight'):
        if w < minorgEdgeWeight:
            G.remove_edge(u, v)
    return G


def nodeDegree_filter(G, nodeDegree_thred):
    """
    node with degree below nodeDegree_thred will be removed
    :param G: networkx G
    :param nodeDegree_thred: minimum node degree
    :return: G
    """
    # remove node based on node degree threshold
    for n in G.nodes():
        if G.degree(n) < nodeDegree_thred:
            G.remove_node(n)
    print('finish remove node: ', time.strftime('%Y-%m-%d %H:%M:%S'))
    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()), '\n')

    return G


def addNode_degree(G):
    """
    Add strength, general and specific degree of nodes
    :param G: directed or undirected Graph
    :return: G
    """
    # add strength degree
    for n in G.nodes():
        G.node[n]['N'] = G.degree(n, weight='weight')
        G.node[n]['n'] = float(G.degree(n))
        if G.node[n]['N'] <= 0.0:
            raise TypeError('find isolated node, or negative weight:{}-{}'.format(n, G.node[n]['label']))
    # --------------
    if nx.is_directed(G):
        G_nei_iter = PF.genChain(G.successors_iter, G.predecessors_iter)
    else:
        G_nei_iter = G.neighbors_iter

    def getMaxMinStrength():
        node = G.nodes_iter().next()
        max_S = G.node[node]['N']
        min_S = G.node[node]['N']
        for n in G.nodes_iter():
            s = G.node[n]['N']
            if s > max_S:
                max_S = s
            if s < min_S:
                min_S = s
        return max_S, min_S

    def getNeiStrength(x):
        s = 0
        for n in G_nei_iter(x):
            s = s + G.node[n]['N']
        return s

    maxS, minS = getMaxMinStrength()
    arrayForpercentile = [G.node[n]['N'] for n in G.nodes()] + [maxS]
    percdict = PF.listtopercentiles(arrayForpercentile)

    # calculate general and specific degree
    for n in G.nodes():
        strength = G.node[n]['N']
        # general degree
        G.node[n]['G_r'] = strength / (getNeiStrength(n) + 0.1)
        G.node[n]['G_n'] = PF.scaling(maxS + 0.1, minS - 0.1, strength)
        G.node[n]['G_p'] = percdict[strength]
        # specific degree
        G.node[n]['SP_r'] = 1.0 - G.node[n]['G_r']
        G.node[n]['SP_n'] = 1.0 - G.node[n]['G_n']
        G.node[n]['SP_p'] = 1.0 - G.node[n]['G_p']
    return G


def addEdge_distance(G, DisTypes, normalization_methods, meanmethods):
    """
    add the general, specific, relevance, combination and other distance of edges
    The distance types in DisTypes are added.
    :param G: networkx directed or undirected graph
    :param DisTypes: the types of distance to be added in edges. ['G','SP','R','C','c','OTHER']
    :param normalization_methods: normalization methods to be tried for node degree and edge distance. ['r','n','p']
    :param meanmethods: avaerage methods to be tried for edge distance. ['AM','GM','HM']
    :return: G
    """

    def getMaxMinWeights():
        max_W = G.edges_iter(data='weight').next()[2]
        min_W = G.edges_iter(data='weight').next()[2]
        for (a, b, w) in G.edges_iter(data='weight'):
            if w > max_W:
                max_W = w
            if w < min_W:
                min_W = w
        return max_W, min_W

    def deg_AM_dist(Da, Db):
        dist = 1.0 - (Da + Db) / 2.0
        return dist

    def deg_GM_dist(Da, Db):
        dist = fabs(log(sqrt(Da * Db)))
        return dist

    def deg_HM_dist(Da, Db):
        dist = (Da + Db) / (2.0 * Da * Db)
        return dist

    maxW, minW = getMaxMinWeights()
    arrayw = [ed[2] for ed in G.edges_iter(data='weight')]
    percdict = PF.listtopercentiles(arrayw)
    deg_To_dist = {'AM': deg_AM_dist, 'GM': deg_GM_dist, 'HM': deg_HM_dist}

    class nodeDe(object):
        def __init__(self, n, w, normalization_methods):
            self.strength = G.node[n]['N']
            self.degree = G.node[n]['n']
            self.n = n
            self.R = {
                'r': w / self.strength,
                'n': PF.scaling(maxW, minW - 0.1, w),
                'p': percdict[w]
            }
            self.G = {
                'r': G.node[n]['G_r'],
                'n': G.node[n]['G_n'],
                'p': G.node[n]['G_p']
            }
            self.SP = {
                'r': G.node[n]['SP_r'],
                'n': G.node[n]['SP_n'],
                'p': G.node[n]['SP_p']
            }
            self.C = {}
            self.c = {}
            for rnm in normalization_methods:
                for gsp_nm in normalization_methods:
                    self.C[rnm + gsp_nm] = self.R[rnm] * self.G[gsp_nm]
                    self.c[rnm + gsp_nm] = self.R[rnm] * self.SP[gsp_nm]
            return

    class edgeDe(object):
        def __init__(self, Anode, Bnode, w):
            self.weight = w
            self.A = Anode
            self.B = Bnode

        def add_edgedist(self, dis_types, normalization_methods, meanmethods):
            for nm in normalization_methods:
                for meanmd in meanmethods:
                    # general distance
                    if 'G' in dis_types:
                        G[self.A.n][self.B.n]['G_{}_{}'.format(nm, meanmd)] = deg_To_dist[meanmd](self.A.G[nm],
                                                                                                  self.B.G[nm])
                    # specific distance
                    if 'SP' in dis_types:
                        G[self.A.n][self.B.n]['SP_{}_{}'.format(nm, meanmd)] = deg_To_dist[meanmd](self.A.SP[nm],
                                                                                                   self.B.SP[nm])
                    # relevance distance
                    if 'R' in dis_types:
                        G[self.A.n][self.B.n]['R_{}_{}'.format(nm, meanmd)] = deg_To_dist[meanmd](self.A.R[nm],
                                                                                                  self.B.R[nm])
            for nmR in normalization_methods:
                for nmGSP in normalization_methods:
                    for meanmd in meanmethods:
                        # combine relevance and general
                        if 'C' in dis_types:
                            G[self.A.n][self.B.n]['C_{}{}_{}'.format(nmR, nmGSP, meanmd)] = deg_To_dist[meanmd](
                                self.A.C[nmR + nmGSP], self.B.C[nmR + nmGSP])
                        # combine relevance and specific
                        if 'c' in dis_types:
                            G[self.A.n][self.B.n]['c_{}{}_{}'.format(nmR, nmGSP, meanmd)] = deg_To_dist[meanmd](
                                self.A.c[nmR + nmGSP], self.B.c[nmR + nmGSP])
            return

        def otherdist(self):
            sqrtn = sqrt(self.A.degree * self.B.degree)
            maxN = max(self.A.strength, self.B.strength)
            minN = min(self.A.strength, self.B.strength)
            for GPer in [0.0, 0.25, 0.5, 0.75, 1.0]:
                d1 = math.pow(maxN, GPer)
                d2 = math.pow(minN, (1.0 - GPer))
                G[self.A.n][self.B.n]['Fd_{}'.format(GPer)] = fabs(log(self.weight / (d1 * d2)))
            G[self.A.n][self.B.n]['Nd'] = fabs(log(1.0 / sqrtn))
            G[self.A.n][self.B.n]['FNd'] = (G[self.A.n][self.B.n]['Nd'] + G[self.A.n][self.B.n]['Fd_0.5']) / 2.0
            return

    for (a, b, w) in G.edges_iter(data='weight'):
        nodeA = nodeDe(a, w, normalization_methods)
        nodeB = nodeDe(b, w, normalization_methods)
        edgeE = edgeDe(nodeA, nodeB, w)
        edgeE.add_edgedist(DisTypes, normalization_methods, meanmethods)
        if 'OTHER' in DisTypes:
            edgeE.otherdist()

    return G


def G2S_direction(G):
    """
    update the relationship direction as 'general --> specifi' if the graph is directed.
    :param G: networkx Graph
    :return: G
    """
    # update direction General-->Specific
    if nx.is_directed(G):
        for (a, b) in G.edges():
            if G.node[a]['N'] < G.node[b]['N']:
                G.add_edge(b, a, G[a][b])
                G.remove_edge(a, b)
        print('finish update direction', time.strftime('%Y-%m-%d %H:%M:%S'))
        print('edges: ', len(G.edges()))
        print('nodes: ', len(G.nodes()), '\n')
    else:
        print('Not directed Graph. No need to update direction')
    return G
