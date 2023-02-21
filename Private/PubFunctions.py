# my common public functions for this project
from math import sqrt, log, fabs
import MySQLdb
import MySQLdb.cursors as cursors
import networkx as nx
import mysql.connector
import pymysql


################################ loadwhole from mysql table or csv ##############################################

# generate graph in one of the three forms: undirected graph, one-directed graph, bi-directed graph
# Checked OK
# reltable refer to all_w2w table in MySQL
def loadw2wdict(schema, reltable, fm):
    assert fm in ['undirected', 'one-directed', 'bi-directed'], 'unknown graph form'

    connection = MySQLdb.connect(host='localhost', user='root', passwd='19900708', db=schema,
                                 cursorclass=cursors.SSCursor)
    cursor = connection.cursor()
    query = 'select rowid,colid,value from {} where rowid<>colid'.format(reltable)
    cursor.execute(query)

    if fm == 'undirected':
        G = nx.Graph()
        G.add_edges_from([(int(n[0]), int(n[1]), {'weight': float(n[2])}) for n in cursor])

    if fm == 'one-directed':
        G = nx.DiGraph()
        G.add_edges_from([(int(n[0]), int(n[1]), {'weight': float(n[2])}) for n in cursor])

    if fm == 'bi-directed':
        G = nx.DiGraph()
        for n in cursor:
            G.add_edge(int(n[0]), int(n[1]), {'weight': float(n[2])})
            G.add_edge(int(n[1]), int(n[0]), {'weight': float(n[2])})

    cursor.close()
    connection.close()

    return G


# update the existing graph by adding node label
# Checked OK
# labtable refer to all_keywords table in MySQL
def load_nodelabel(G, schema, labtable):
    connection = MySQLdb.connect(host='localhost', user='root', passwd='19900708', db=schema,
                                 cursorclass=cursors.SSCursor, charset='utf8')
    cursor = connection.cursor()
    query = "select id,word from {}".format(labtable)
    cursor.execute(query)

    G.add_nodes_from([(int(n[0]), {'label': n[1]}) for n in cursor])

    cursor.close()
    connection.close()

    return G


# return edgelist
def loadw2wlist(schema, reltable):
    connection = MySQLdb.connect(host='localhost', user='root', passwd='19900708', db=schema,
                                 cursorclass=cursors.SSCursor)
    cursor = connection.cursor()
    query = 'select rowid,colid,value from {} where rowid<>colid'.format(reltable)
    cursor.execute(query)

    edgelist = [(int(n[0]), int(n[1]), {'weight': float(n[2])}) for n in cursor]
    # G=nx.from_edgelist(edgelist)

    # return G
    return edgelist


# return graph
def loadcsv(path):
    f = open(path, 'rb')
    G = nx.read_edgelist(f, nodetype=int, data=[('weight', float)], edgetype=float)
    f.close()

    return G


############################### read neighbors of a local node from MySQL ####################
# all all three algorithms(BFS,DFS,complexDFS) get the same results, For efficiency, BFS better than DFS bettern than complexDFS

# output the cursor of a schema
def creatCursor(shema, type):  # OK no problem!!!

    if type == "W":
        cnx = mysql.connector.connect(user='root', passwd='19900708', database=shema)
    elif type == "R":
        cnx = pymysql.connect(user='root', passwd='19900708', database=shema, charset='utf8')
    else:
        raise TypeError("No Type for Create Cursor")

    cursor = cnx.cursor()

    return cnx, cursor


# save the edges of a node into graph, and return direct neighbors set around a node
def SaveEdge_ReturnAdjacent(id, sG, rcursor, reltable):
    Qr = "select rowid,colid,value from {} where (rowid={} or colid={}) and (rowid!=colid)".format(reltable, id, id)
    rcursor.execute(Qr)
    sG.add_edges_from([(n[0], n[1], {'weight': float(n[2])}) for n in rcursor])
    return set(sG.neighbors(id))


# read neighbors of a node within maxlevel level (recursive)
# All the edges of nodes(whose level<=maxlevel) are saved
# if maxlevel default to -1, find whole connected part
def DFS_recursive_LevelNeighbor(id, sG, rcursor, reltable, finished=None, l=0, maxlevel=-1):
    if finished is None:
        finished = set()
    nb = SaveEdge_ReturnAdjacent(id, sG, rcursor, reltable)
    finished.add(id)
    print('Node:{}, level:{}'.format(id, l))

    if l == maxlevel:
        return
    else:
        for node in nb - finished:
            # if node in nb-finished:
            DFS_recursive_LevelNeighbor(node, sG, rcursor, reltable, finished, l + 1, maxlevel)
        return


# read neighbors of a node within maxlevel level (BFS)
def BFS_LevelNeighbor(id, sG, rcursor, reltable, l=0, maxlevel=-1):
    finished = set()
    nb = SaveEdge_ReturnAdjacent(id, sG, rcursor, reltable)
    finished.add(id)
    print('Node:{},level:{}'.format(id, l))

    while l != maxlevel:
        if nb - finished:
            l = l + 1
            queue = nb - finished
            nb = set()
            for node in queue:
                nb.update(SaveEdge_ReturnAdjacent(node, sG, rcursor, reltable))
                finished.add(node)
                print('Node:{},level:{}'.format(node, l))
        else:
            break

    return


# read neighbors of a node within maxlevel level (DFS)
def Complex_DFS_LevelNeighbor(id, sG, rcursor, reltable, l, finished=None):
    if finished is None:
        finished = {}
    if l == 0:
        if id in finished:
            if 0 not in finished[id]:
                finished[id][0] = SaveEdge_ReturnAdjacent(id, sG, rcursor, reltable)
                print('Node:{}, Level:{}'.format(id, l))
        else:
            finished[id] = {0: SaveEdge_ReturnAdjacent(id, sG, rcursor, reltable)}
            print('Node:{}, Level:{}'.format(id, l))

        return finished[id][0]
    else:
        if id in finished:
            if l not in finished[id]:
                finished[id][l] = set()
                for node in Complex_DFS_LevelNeighbor(id, sG, rcursor, reltable, 0, finished):
                    finished[id][l].update(Complex_DFS_LevelNeighbor(node, sG, rcursor, reltable, l - 1, finished))
        else:
            finished[id] = {l: set()}
            for node in Complex_DFS_LevelNeighbor(id, sG, rcursor, reltable, 0, finished):
                finished[id][l].update(Complex_DFS_LevelNeighbor(node, sG, rcursor, reltable, l - 1, finished))

        return finished[id][l]


############################ update weights and domain dissimilarity of a graph ################

# calculate one of three directed weights of an edge in any form of graph
# Checked OK
def diweight(G, a, b, w, tp, fm):
    assert fm in ['bi-directed', 'undirected', 'one-directed'], 'the form of graph is not known'

    if fm == 'bi-directed':
        assert type(G) == nx.classes.digraph.DiGraph, "not directed G"
        assert G.number_of_edges(a, b) + G.number_of_edges(b, a) == 2, "not bi-directed G"
        # w=G[a][b]['weight']
        Na = G.out_degree(a, weight='weight')
        na = float(G.out_degree(a))
    else:
        if fm == 'undirected':
            assert type(G) == nx.classes.graph.Graph, "not undirected G"
        if fm == 'one-directed':
            assert type(G) == nx.classes.digraph.DiGraph, "not directed G"
            assert G.number_of_edges(a, b) + G.number_of_edges(b, a) == 1, "not one-directed G"
        # w=G[a][b]['weight']
        Na = G.degree(a, weight='weight')
        na = float(G.degree(a))

    if tp == 'Fw':
        return w / Na
    if tp == 'Nw':
        return 1.0 / na
    if tp == 'FNw':
        return sqrt(w / (Na * na))


# update graph with 3 undirected weight, the graph is undirected graph or one-directed graph
# cheked OK
def unweight(G, a, b, w):
    Na = G.degree(a, weight='weight')
    na = float(G.degree(a))
    Nb = G.degree(b, weight='weight')
    nb = float(G.degree(b))

    G[a][b]['Fw'] = round(fabs(log(w / (sqrt(Na * Nb)))), 2)
    G[a][b]['Nw'] = round(fabs(log(1.0 / (sqrt(na * nb)))), 2)
    G[a][b]['FNw'] = round((G[a][b]['Fw'] + G[a][b]['Nw']) / 2.0, 2)

    return


# update G with domain dissimilarity, the graph is undirected graph or one-directed graph
# checked OK
def domain_dis(G, a, b, fm):
    assert fm in ['undirected', 'one-directed'], 'the form of graph is not known'

    if fm == 'undirected':
        Nei_a = set(G.neighbors(a))
        Nei_b = set(G.neighbors(b))
    if fm == 'one-directed':
        Nei_a = set(G.successors(a)).union(set(G.predecessors(a)))
        Nei_b = set(G.successors(b)).union(set(G.predecessors(b)))

    Dis = 1.0 - (float(len(Nei_a.intersection(Nei_b)))) / (float(len(Nei_a.union(Nei_b))))
    G[a][b]['dis'] = round(fabs(log(Dis)), 2)

    return


# update undirected graph with three undirected weights and domain dissimilarity. result undirected graph with undirected weights
# checked OK
def uG_to_uGuW(G, fm='undirected'):
    assert type(G) == nx.classes.graph.Graph, "not undirected G"

    for (a, b, w) in G.edges_iter(data='weight'):
        # three undirected weights
        unweight(G, a, b, w)

        # domain dissimilarity
        domain_dis(G, a, b, fm)

    return G


# update one-directed graph with undirected weight, domain dissimilarity and new direction. The new direction is G2S
# checked OK
def dG_to_dGuW(dG, tp, fm='one-directed'):
    assert type(dG) == nx.classes.digraph.DiGraph, "not directed Graph"

    for (a, b, w) in dG.edges(data='weight'):
        if diweight(dG, a, b, w, tp, fm) <= diweight(dG, b, a, w, tp, fm):
            unweight(dG, a, b, w)
            domain_dis(dG, a, b, fm)
        else:
            dG.add_edge(b, a, dG[a][b])
            dG.remove_edge(a, b)
            unweight(dG, b, a, w)
            domain_dis(dG, b, a, fm)

    return dG


# update bi-directed graph with directed weights
# checked OK
def bdG_to_bdGdW(bdG, tp, fm='bi-directed'):
    assert type(bdG) == nx.classes.digraph.DiGraph, "not directed Graph"

    for (a, b, w) in bdG.edges_iter(data='weight'):
        logWab = round(fabs(log(diweight(bdG, a, b, w, tp, fm))), 2)
        logWba = round(fabs(log(diweight(bdG, b, a, w, tp, fm))), 2)

        bdG[a][b]['Forward'] = logWab
        bdG[a][b]['Backward'] = logWba

    return bdG
