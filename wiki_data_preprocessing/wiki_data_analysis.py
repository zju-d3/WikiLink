# to write the w2wdict
# import mysql.connector
# import MySQLdb
import networkx as nx
# import pymysql
import datetime
import time
from math import sqrt, log, fabs
import math
import pandas as pd
import PF
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
#
# def creatCursor(shema, type):  # OK no problem!!!
#
#     if type == "W":
#         cnx = mysql.connector.connect(user='root', password='root', database=shema)
#     elif type == "R":
#         cnx = pymysql.connect(user='rd', password='xx', database=shema, charset='utf8')
#     else:
#         raise TypeError("No Type for Create Cursor")
#
#     cursor = cnx.cursor()
#
#     return cnx, cursor


def handle_w2w(Cursor, ids, table):
    starttime = time.clock()

    sids = sorted(ids)

    x = len(ids)

    for i in range(0, x - 1):
        for j in range(i + 1, x):
            RQy = ("""
            select EXISTS(SELECT rowid FROM {} WHERE rowid={} and colid={})
            """.format(table, sids[i], sids[j]))
            Cursor.execute(RQy)
            ex = Cursor.fetchone()[0]

            if ex == 0:
                WQy = ("""
                insert into {} values({},{},1)
                """.format(table, sids[i], sids[j]))

            else:
                WQy = ("""
                update
                `{}`
                set
                `value`=`value`+1

                WHERE rowid={} and colid={}
                """.format(table, sids[i], sids[j]))

            Cursor.execute(WQy)

    endtime = time.clock()
    if x > 5:
        print('length:{} , time: {} s'.format(x, endtime - starttime))
    return


# Cursor,
# ids is the list of co-occurence
# table is the name in the schema
# journal, no need. modify.

def loadw2wdict_csv(fm, path):
    assert fm in ['undirected', 'one-directed', 'bi-directed'], 'unknown graph form'
    df = pd.read_csv(path, sep=';', names=['h', 't', 'v'])
    if fm == 'undirected':
        G = nx.Graph()
        for tup in zip(df['h'], df['t'], df['v']):
            # print ('1 {} 2 {} 3 {}'.format(tup[0],tup[1],tup[2]))

            G.add_edge(tup[0], tup[1], attr_dict={'weight': tup[2]})
            if tup[2] <= 0:
                print('1 {} 2 {} 3 {}'.format(tup[0], tup[1], tup[2]))
    return G

def filter_from_csv(path,node_degree,edges_number):
    df= pd.read_csv(path,sep=',',names =['head','tail','number'])
    print (df)
    # df_new =df[df['number']>edges_number]
    # df_new['head'].value_counts()
    # df_new['tail'].value_counts()
    #

#
# def loadw2wdict(schema, reltable, fm):
#     assert fm in ['undirected', 'one-directed', 'bi-directed'], 'unknown graph form'
#
#     connection = MySQLdb.connect(host='localhost', user='rd', passwd='B-Link2020', db=schema,
#                                  cursorclass=cursors.SSCursor)
#     cursor = connection.cursor()
#     query = 'select rowid,colid,value from {} where rowid<>colid'.format(reltable)
#     cursor.execute(query)
#
#     if fm == 'undirected':
#         G = nx.Graph()
#         G.add_edges_from([(int(n[0]), int(n[1]), {'weight': float(n[2])}) for n in cursor])
#
#     if fm == 'one-directed':
#         G = nx.DiGraph()
#         G.add_edges_from([(int(n[0]), int(n[1]), {'weight': float(n[2])}) for n in cursor])
#
#     if fm == 'bi-directed':
#         G = nx.DiGraph()
#         for n in cursor:
#             G.add_edge(int(n[0]), int(n[1]), {'weight': float(n[2])})
#             G.add_edge(int(n[1]), int(n[0]), {'weight': float(n[2])})
#
#     cursor.close()
#     connection.close()
#
#     return G
#
#
# def loadw2wdict(schema, reltable, fm):
#     assert fm in ['undirected', 'one-directed', 'bi-directed'], 'unknown graph form'
#
#     connection = MySQLdb.connect(host='localhost', user='root', passwd='root', db=schema)
#     cursor = connection.cursor()
#     query = 'select rowid,colid,value from {} where rowid<>colid'.format(reltable)
#     cursor.execute(query)
#
#     if fm == 'undirected':
#         G = nx.Graph()
#         G.add_edges_from([(int(n[0]), int(n[1]), {'weight': float(n[2])}) for n in cursor])
#
#     if fm == 'one-directed':
#         G = nx.DiGraph()
#         G.add_edges_from([(int(n[0]), int(n[1]), {'weight': float(n[2])}) for n in cursor])
#
#     if fm == 'bi-directed':
#         G = nx.DiGraph()
#         for n in cursor:
#             G.add_edge(int(n[0]), int(n[1]), {'weight': float(n[2])})
#             G.add_edge(int(n[1]), int(n[0]), {'weight': float(n[2])})
#
#     cursor.close()
#     connection.close()
#
#     return G


def uG_to_uGuW(G, fm='undirected'):
    assert type(G) == nx.classes.graph.Graph, "not undirected G"

    for (a, b, w) in G.edges_iter(data='weight'):
        # three undirected weights
        unweight(G, a, b, w)

        # domain dissimilarity
        domain_dis(G, a, b, fm)

    return G


def unweight(G, a, b, w):
    """
    Similar to unweight_allocation below
    The only difference is that
    in this function, all the weight put equal pow (half and half) on the two nodes of the edge.
    :param G:
    :param a:
    :param b:
    :param w:
    :return:
    """

    Na = G.degree(a, weight='weight')
    na = float(G.degree(a))
    Nb = G.degree(b, weight='weight')
    nb = float(G.degree(b))

    G[a][b]['Fw'] = round(fabs(log(w / (sqrt(Na * Nb)))), 2)
    G[a][b]['Nw'] = round(fabs(log(1.0 / (sqrt(na * nb)))), 2)
    G[a][b]['FNw'] = round((G[a][b]['Fw'] + G[a][b]['Nw']) / 2.0, 2)

    return


def domain_dis(G, a, b, fm):
    assert fm in ['undirected', 'one-directed'], 'the form of graph is not known'

    if fm == 'undirected':
        Nei_a = set(G.neighbors(a))
        Nei_b = set(G.neighbors(b))
    if fm == 'one-directed':
        Nei_a = set(G.successors(a)).union(set(G.predecessors(a)))
        Nei_b = set(G.successors(b)).union(set(G.predecessors(b)))

    Dis = 1.0 - (float(len(Nei_a.intersection(Nei_b)))) / (float(len(Nei_a.union(Nei_b))))
    print(Dis)
    G[a][b]['dis'] = round(fabs(log(Dis)), 2)

    return
#
#
# def write_undirected(schema, reltable, labtable):
#     uG = loadw2wdict(schema, reltable, 'undirected')
#     print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': undirected graph readed')
#
#     uG = uG_to_uGuW(uG, 'undirected')
#     print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': undirected graph add weights')
#
#     uG = PubFunctions.load_nodelabel(uG, schema, labtable)
#     print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': undirected graph add node label')
#
#     nx.write_gpickle(uG, 'undirected.gpickle')
#     print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), ': undirected graph written')
#
#     return
#
#
# def load_nodelabel(G, schema, labtable):
#     connection = MySQLdb.connect(host='localhost', user='rd', passwd='B-Link2020', db=schema,
#                                  cursorclass=cursors.SSCursor, charset='utf8')
#     cursor = connection.cursor()
#     query = "select id,word from {}".format(labtable)
#     cursor.execute(query)
#
#     G.add_nodes_from([(int(n[0]), {'label': n[1]}) for n in cursor])
#
#     cursor.close()
#     connection.close()
#
#     return G


def unweight_allocation(G, a, b, w, Lp, Sp):
    """
    Assign three undirected weight to the graph.
    FNw is the mix of frequence weight and degree weight, half Fw and half Nw
    Nw is the degree weight, half and half pow for the two nodes
    Fw is frequence weight, Lp is the pow of node with large frequence, Sp is the pow of node with small frequence.
    :param G: THe nx graph
    :param a: one end of the edge
    :param b: the other end of the edge
    :param w: w is the original weight of the edge
    :param Lp: Only For Fw, Lp is the power percentage allocated to the large frequent node.
    :param Sp: Only For Fw, Sp is the power percentage allocated to the small frequent node.
    :return:
    """

    Na = G.degree(a, weight='weight')
    na = float(G.degree(a))
    Nb = G.degree(b, weight='weight')
    nb = float(G.degree(b))
    maxN = float(max(Na, Nb))
    minN = float(min(Na, Nb))
    d1 = math.pow(maxN, Lp)
    d2 = math.pow(minN, Sp)

    G[a][b]['Fw'] = round(fabs(log(w / (d1 * d2))), 2)
    G[a][b]['Nw'] = round(fabs(log(1.0 / (sqrt(na * nb)))), 2)
    G[a][b]['FNw'] = round((G[a][b]['Fw'] + G[a][b]['Nw']) / 2.0, 2)

    return


def reduceGraph_new(read_g, write_g, minEdgeWeight, minNodeDegree):
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

    print('update weight ok')
    nx.write_gpickle(G, write_g)

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
    count = 0
    for (a, b, w) in G.edges_iter(data='weight'):
        count = count + 1
        unweight_allocation(G, a, b, w, Lp, Sp)
        print(count)
    print('update weight ok')
    nx.write_gpickle(G, write_g)

    return


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
    count = 0

    for (a, b, w) in tqdm(G.edges(data='weight')):
        # if count>=100 and count% 100==0 and (count/100) %2!=0:
        #     start = time.clock()
        maxalpha, minalpha = get_maxAlpha(G, a, b, w)
        G[a][b]['maxAlpha'] = maxalpha
        G[a][b]['minAlpha'] = minalpha
        count = count + 1
        #
        # if count>100 and  count % 100 == 0 and (count / 100) % 2 == 0:
        #     end = time.clock()
        #     print (end-start)
    print('finish calculate alpha', time.strftime('%Y-%m-%d %H:%M:%S'))

    print('edges: ', len(G.edges()))
    print('nodes: ', len(G.nodes()), '\n')

    return G



def disparity_filter_sectional(G):
    """
    wiki network reduction by disparity filter, a sectional function to reduce node with lower degree not too much,
    while reduce node with higher degree severe.

    :param G:  the wiki network
    :return:
    """

    def return_max_and_node(a, b):

        if G.degree(a) > G.degree(b):
            return G.degree(a), a

        if G.degree(a) <= G.degree(b):
            return G.degree(b), b

    def alpha_thred_sectional_function(a,b):
        degree=max(G.degree(a),G.degree(b))
        if 1<=degree<10:
            alpha_thred =0.3
        if 10<=degree<100:
            alpha_thred =0.4
        if 100<=degree<1000:
            alpha_thred =0.60
        if 1000<=degree<10000:
            alpha_thred =0.65
        if 10000<=degree<15000:
            alpha_thred=0.75
        if 15000<=degree<40000:
            alpha_thred=0.90
        if 40000<=degree<60000:

            alpha_thred=0.95
        if 60000<=degree<81000:
            alpha_thred=0.98
        return alpha_thred
    for (a, b) in tqdm(G.edges()):
        if G[a][b]['maxAlpha'] < alpha_thred_sectional_function(a,b):
            G.remove_edge(a, b)
    print('finish filter', time.strftime('%Y-%m-%d %H:%M:%S'))
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

    for (a, b) in tqdm(G.edges()):
        if G[a][b]['maxAlpha'] < alpha_thred:
            G.remove_edge(a, b)
    print('finish filter', time.strftime('%Y-%m-%d %H:%M:%S'))
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
        if G.degree(n, weight='weight') == 0:
            G.remove_node(n)
            continue
        G.node[n]['N'] = G.degree(n, weight='weight')
        G.node[n]['n'] = float(G.degree(n))

        if G.node[n]['N'] <= 0.0:
            # raise TypeError('find isolated node, or negative weight:{}-{}'.format(n, G.node[n]['label']))
            raise TypeError('find isolated node, or negative weight:{}-{}'.format(n, 'test'))
            # modify 0616, why isolated?
    # --------------
    if nx.is_directed(G):
        G_nei_iter = PF.genChain(G.successors_iter, G.predecessors_iter)
    else:
        G_nei_iter = G.neighbors_iter

    def getMaxMinStrength():
        node = G.nodes_iter().__next__()
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
    for n in tqdm(G.nodes()):
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
        max_W = G.edges_iter(data='weight').__next__()[2]
        min_W = G.edges_iter(data='weight').__next__()[2]
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
