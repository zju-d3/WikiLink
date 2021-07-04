import networkx as nx
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.utils.graph import graph_laplacian
from sklearn.utils.arpack import eigsh
from sklearn.cluster.k_means_ import k_means
import math
from heapq import heappush, heappop
import time
import networkx as nx
from networkanalysis.Process_OriginalTable import main

from networkanalysis.Analysis import Retrievor

class tri(object):
    def print_parameters(self,parameters,function):
        function(**parameters)

    def lala(self,a,b):
        print a,b

def original_cluster(k):
    G = nx.read_gpickle('data/undirected(fortest).gpickle')
    A = nx.adjacency_matrix(G,nodelist=G.nodes()[:-1],weight='weight')
    labels=spectral_clustering(A,n_clusters=k,eigen_solver='arpack')
    return labels

def my_uniteigenvector_zeroeigenvalue_cluster(k):
    G = nx.read_gpickle('data/undirected(fortest).gpickle')
    A = nx.adjacency_matrix(G, nodelist=G.nodes()[:-1], weight='weight')
    #A=A.toarray()
    #np.fill_diagonal(A,0.01) #add node with its own weight to itself
    #Tri = np.diag(np.sum(A, axis=1))
    #L = Tri - A
    #Tri_1 = np.diag(np.reciprocal(np.sqrt(Tri).diagonal()))
    #Ls = Tri_1.dot(L).dot(Tri_1)

    Ls, dd = graph_laplacian(A,normed=True, return_diag=True)

    eigenvalue_n, eigenvector_n = eigsh(Ls*(-1), k=k,
                                   sigma=1.0, which='LM',
                                   tol=0.0)

    #for ic,vl in enumerate(eigenvalue_n):
    #    if abs(vl-0)<=1e-10:
    #        eigenvector_n[:, ic] = np.full(len(G.nodes()[:-1]),1.0 / math.sqrt(len(G.nodes()[:-1]))) # zero eigenvalue

    eigenvector_n[:, -1] = np.full(len(G.nodes()[:-1]), 1.0 / math.sqrt(len(G.nodes()[:-1])))  # zero eigenvalue

    for ir,n in enumerate(eigenvector_n):
        eigenvector_n[ir]=n/float(np.linalg.norm(n))  # normalize to unitvector

    _, labels, _ = k_means(eigenvector_n, k, random_state=None,
                           n_init=100)
    return labels

def cluster_G():
    G=nx.Graph()

    G.add_edge('A','B',weight=1)
    G.add_edge('A', 'C', weight=1)
    G.add_edge('C', 'B', weight=1)

    G.add_edge('X', 'Y', weight=1)
    G.add_edge('X', 'Z', weight=1)
    G.add_edge('Z', 'Y', weight=1)

    G.add_edge('A', 'X', weight=1)
    G.add_edge('A', 'E', weight=1)
    G.add_edge('E', 'X', weight=1)
    G.add_edge('A', 'F', weight=1)
    G.add_edge('F', 'X', weight=1)
    G.add_edge('E', 'F', weight=0.1)

    G.add_edge('C', 'Z', weight=1.5)
    G.add_edge('C', 'Y', weight=2.5)

    G.add_edge('B', 'Y', weight=3)

    G.add_edge('A', 'G', weight=0.5)
    G.add_edge('G', 'F', weight=0.6)

    return G


def dijkstra_cluster(G,cluster1,cluster2,tp):
    cset1=set(cluster1)
    cset2=set(cluster2)
    push=heappush
    pop=heappop

    fringe = []

    for n in cluster1:
        push(fringe,(0,[n]))


    while fringe:
        (d,p)=pop(fringe)
        end=p[-1]

        if end in cset2:  # reach cluster2
            yield (d,p)
            continue

        for nei in G.adj[end].keys():
            if nei not in cset1 and nei not in p:
                up_d=d+G[nei][end][tp]
                up_p=p+[nei]
                push(fringe,(up_d,up_p))




def test_clusterPaths():
    R = Retrievor.UndirectedG(nx.read_gpickle('../undirected(abcdeijm_test).gpickle'), 'abcdeijm_test')
    cluster1=[R.G.neighbors(200)[0]]+[200]
    cluster2=[R.G.neighbors(20000)[0]]+[20000]
    cset1=set(cluster1)
    cset2=set(cluster2)
    print 'cluster1:',cluster1
    print 'cluster2:', cluster2
    q_pair=[]
    for s in cluster1:
        for t in cluster2:
            ge=R.get_pathsBetween_twonodes(s,t,'Fw',1)
            length, path=ge.next()
            while ( cset1.issubset(set(path)) or cset2.issubset(set(path)) ):
                length, path = ge.next()
            q_pair.append((length,path))

    q_pair=sorted(q_pair,key=lambda x:x[0])
    final_length=q_pair[-1][0]+0.001
    all_pair=[]

    for (length,path) in q_pair:
        ge=R.get_pathsBetween_twonodes(path[0],path[-1],'Fw',1)
        while True:
            d,p=ge.next()
            while ( cset1.issubset(set(p)) or cset2.issubset(set(p)) ):
                d,p=ge.next()
            if d > final_length:
                break
            else:
                all_pair.append((d,p))
                print 'all_pair',d,p


    all_pair=sorted(all_pair,key=lambda x:x[0])

    cl_paths=[]
    ge=R.get_pathsBetween_twoClusters(cluster1,cluster2,'Fw')
    while True:
        d,p=ge.next()
        if d > final_length:
            break
        else:
            cl_paths.append((d,p))
            print 'cl_paths',d,p


    print 'all_pair:',all_pair
    print 'cl_paths:',cl_paths
    if len(all_pair)==len(cl_paths):
        all_length = [round(n[0],2) for n in all_pair]
        cl_length = [round(n[0],2) for n in cl_paths]
        if all_length == cl_length:
            print 'successfull!!!!!!!!'

    print 'finish'
    return



def test_normalizedCluster():
    G = nx.Graph()
    G.add_edge('a','b',weight=1.0)
    G.add_edge('a', 'd', weight=2.0)
    G.add_edge('a', 'c', weight=3.0)
    G.add_edge('b', 'd', weight=4.0)
    G.add_edge('b', 'c', weight=5.0)
    G.add_edge('c', 'e', weight=6.0)
    G.add_edge('f','g',weight = 5.0)
    G.add_edge('h', 'g', weight=1.0)
    G.add_edge('h', 'f', weight=1.0)


    """G.add_edge(1, 2, weight=1.0)
    G.add_edge(1, 4, weight=2.0)
    G.add_edge(1, 3, weight=3.0)
    G.add_edge(2, 4, weight=4.0)
    G.add_edge(2, 3, weight=5.0)
    G.add_edge(3, 5, weight=6.0)
    G.add_node(6)
    G.add_node(7)
    G.add_node(8)
    G.add_edge(6, 7, weight=5.0)
    G.add_edge(8, 6, weight=1.0)
    G.add_edge(8, 7, weight=1.0)"""

    #G1 = nx.Graph()
    #G1.add_edge('a', 'b', weight=2)
    #G1.add_edge('a', 'c', weight=1)
    #G1.add_edge('b', 'c', weight=1)
    #G1.add_node('d')

    R = Retrievor.UndirectedG(G, 'fortest')

    print '5 nodes'
    for k in [1,2,3,4]:
        clustersLs = R.cutgraph(['a', 'b', 'c', 'd', 'e'], k, Mx='Ls')
        clustersLa = R.cutgraph(['a', 'b', 'c', 'd', 'e'], k, Mx='La')
        clustersLsLa = R.cutgraph(['a', 'b', 'c', 'd', 'e'], k, Mx='LsLa')
        print 'clustersLs'
        print clustersLs
        print 'clustersLa'
        print clustersLa
        print 'clustersLsLa'
        print clustersLsLa

    print ' 6 nodes'
    for k in [1,2, 3, 4, 5]:
        clustersLs = R.cutgraph(['a', 'b', 'c', 'd', 'e', 'f'], k, Mx='Ls')
        clustersLa = R.cutgraph(['a', 'b', 'c', 'd', 'e', 'f'], k, Mx='La')
        clustersLsLa = R.cutgraph(['a', 'b', 'c', 'd', 'e','f'], k, Mx='LsLa')
        print 'clustersLs'
        print clustersLs
        print 'clustersLa'
        print clustersLa
        print 'clustersLsLa'
        print clustersLsLa

    print '7 nodes'
    for k in [1,2, 3, 4, 5, 6]:
        clustersLs = R.cutgraph(['a', 'b', 'c', 'd', 'e','f','g'], k, Mx='Ls')
        clustersLa = R.cutgraph(['a', 'b', 'c', 'd', 'e','f','g'], k, Mx='La')
        clustersLsLa = R.cutgraph(['a', 'b', 'c', 'd', 'e','f','g'], k, Mx='LsLa')
        print 'clustersLs'
        print clustersLs
        print 'clustersLa'
        print clustersLa
        print 'clustersLsLa'
        print clustersLsLa

    print ' 8 nodes'
    for k in [1,2, 3, 4, 5, 6, 7]:
        clustersLs = R.cutgraph(['a', 'b', 'c', 'd', 'e','f','g','h'], k, Mx='Ls')
        clustersLa = R.cutgraph(['a', 'b', 'c', 'd', 'e','f','g','h'], k, Mx='La')
        clustersLsLa = R.cutgraph(['a', 'b', 'c', 'd', 'e','f','g','h'], k, Mx='LsLa')
        print 'clustersLs'
        print clustersLs
        print 'clustersLa'
        print clustersLa
        print 'clustersLsLa'
        print clustersLsLa


    """print 'mcl'
    for r in np.linspace(1,15,num=45):
        M,clusters = R.mcl_cluster(['a','b','c','d','e','f','g','h'],r)
        print r,':',clusters"""

    return


def test_clustersCentrality():
    G = nx.Graph()
    G.add_edge('a', 'b', Fw = 2.047)
    G.add_edge('a', 'd', Fw=1.099)
    G.add_edge('a', 'c', Fw=1.117)
    G.add_edge('b', 'd', Fw=0.661)
    G.add_edge('b', 'c', Fw=0.861)
    G.add_edge('c', 'e', Fw=0.424)
    G.add_node('f')

    R = Retrievor.UndirectedG(G, 'fortest')

    print 'For whole graph, cutgraph_sp:'
    for k in [1, 2, 3, 4, 5]:
        clusters = R.cutgraph_sp(['a', 'b', 'c', 'd', 'e'], k)
        clusters = R.sort_clustersCentrality(clusters,'Fw')
        print clusters

    return


def see_variable():
    a=[1]
    def haha():
        b=a[0]



    haha()
    return a


def testCluster():
    G = nx.Graph()

    G.add_edge('a','b',weight = 6.0)
    G.add_edge('b', 'c', weight=5.0)
    G.add_edge('c', 'd', weight=4.0)
    R = Retrievor.UndirectedG(G, 'fortest')
    clustersLs, clustersLa = R.cutgraph(['a', 'b', 'c', 'd'], 3)
    return  clustersLs,clustersLa

def test1():
    G = nx.Graph()
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 4, weight=1.0)
    G.add_edge(4, 1, weight=1.0)
    G.add_edge(2, 4, weight=1.0)
    G.add_edge(1, 6, weight=1.0)
    G.add_edge(4, 5, weight=1.0)
    G.add_edge(3, 7, weight=1.0)
    G.add_edge(5, 6, weight=1.0)
    G.add_edge(5, 7, weight=1.0)
    G.add_edge(6, 7, weight=1.0)

    R = Retrievor.UndirectedG(G, 'fortest')
    clustersLs, clustersLa = R.cutgraph([1,2,3,4,5,6,7],2)

    return clustersLs, clustersLa

def testshortestpath():
    G = nx.Graph()
    G.add_edges_from([('a','e'),('a','d'),('d','e'),('b','d'),('b','c'),('c','e'),('c','d'),('b','e')])
    gen = nx.shortest_simple_paths(G,'a','b')
    for n in gen:
        print n
    print G.nodes()
    print '-----'
    G1 = nx.Graph()
    G1.add_edges_from([('a', 'e'), ('a', 'd'), ('d', 'e'), ('b', 'd'), ('b', 'c'), ('c', 'e'), ('c', 'd'), ('b', 'e'),('d', 'f'),('d', 'g'),('g','f')])
    gen1 = nx.shortest_simple_paths(G1, 'a', 'b')
    for n in gen1:
        print n
    print G1.nodes()


    print '-------'
    G3=nx.read_gpickle('try.gpickle')
    for (a,b) in G3.edges():
        G3[a][b]['Fw'] = int(G3[a][b]['Fw']*10000)

    gen3=nx.shortest_simple_paths(G3,230349,542752,weight='Fw')
    for path in gen3:
        l=0
        for n1,n2 in zip(path,path[1:]):
            l += G3[n1][n2]['Fw']
        print l,path

    print '-------'

    gen3 = nx.shortest_simple_paths(G3, 230349, 542752)
    allp=[]
    for path in gen3:
        l = 0
        for n1, n2 in zip(path, path[1:]):
            l += G3[n1][n2]['Fw']
        allp.append((l,path))
    allp=sorted(allp, key=lambda x:x[0])
    for p in allp:
        print p


def fast_shortestpath(R):
    ta1=time.time()
    length, path=nx.bidirectional_dijkstra(R.G,1,1000,weight='Fw')
    ta2=time.time()
    print 'bidirectional: ',ta2-ta1
    print path

    ta1 = time.time()
    path=nx.dijkstra_path(R.G, 1, 1000, weight='Fw')
    ta2 = time.time()
    print 'single dijkstra_path: ', ta2 - ta1
    print path

    ta1 = time.time()
    gen=R.get_pathsBetween_twonodes(1,1000,'Fw',1)
    (length, path)=gen.next()
    ta2 = time.time()
    print 'my dijkstra_path: ', ta2 - ta1
    print path

def testNodeEdgedegree():
    G = nx.read_gpickle('data/undirected(fortest).gpickle')
    main.nodeDegree_filter(G,1)
    main.addNode_degree(G)
    main.addEdge_distance(G,['G','SP','R','C','c','OTHER'])
    n=3
    print 'node:{}'.format(n)
    node_results = [
        ('key','machine','hand','error'),
        ('label', G.node[n]['label'], 'C', 'None' ),
        ('N', G.node[n]['N'], 14, G.node[n]['N']-14  ),
        ('n', G.node[n]['n'], 3, G.node[n]['n']-3  ),
        ('G_r', G.node[n]['G_r'], 0.633, G.node[n]['G_r']-0.633 ),
        ('G_n', G.node[n]['G_n'], 0.988, G.node[n]['G_n']-0.987 ),
        ('G_p', G.node[n]['G_p'], 0.917, G.node[n]['G_p']-0.917 ),
        ('SP_r', G.node[n]['SP_r'], 0.367, G.node[n]['SP_r']-0.367 ),
        ('SP_n', G.node[n]['SP_n'], 0.012, G.node[n]['SP_n']-0.013 ),
        ('SP_p', G.node[n]['SP_p'], 0.083, G.node[n]['SP_p']-0.083 )
    ]
    for i in node_results:
        print i
    assert len(node_results)-1 == len(G.node[n].keys()), 'key number not mathing'

    m=1
    print 'node:{}'.format(m)
    node_results = [
        ('key', 'machine', 'hand', 'error'),
        ('label', G.node[m]['label'], 'A', 'None'),
        ('N', G.node[m]['N'], 6, G.node[m]['N'] -6 ),
        ('n', G.node[m]['n'], 3, G.node[m]['n'] -3 ),
        ('G_r', G.node[m]['G_r'], 0.199, G.node[m]['G_r'] - 0.199),
        ('G_n', G.node[m]['G_n'], 0.012, G.node[m]['G_n'] - 0.012),
        ('G_p', G.node[m]['G_p'], 0.333, G.node[m]['G_p'] - 0.333),
        ('SP_r', G.node[m]['SP_r'], 0.801, G.node[m]['SP_r'] - 0.801),
        ('SP_n', G.node[m]['SP_n'], 0.988, G.node[m]['SP_n'] - 0.988),
        ('SP_p', G.node[m]['SP_p'], 0.667, G.node[m]['SP_p'] - 0.667)
    ]
    for i in node_results:
        print i
    assert len(node_results) - 1 == len(G.node[m].keys()), 'key number not mathing'

    print 'edges:{}-{}'.format(n,m)
    edge_results =[
        ('key', 'machine', 'hand', 'error'),
        ('G_r_AM', G[n][m]['G_r_AM'], 0.584 ),
        ('G_n_GM', G[n][m]['G_n_GM'], 2.217),
        ('G_p_HM', G[n][m]['G_p_HM'], 2.047),
        ('R_p_AM', G[n][m]['R_p_AM'], 0.5),
        ('R_r_GM', G[n][m]['R_r_GM'], 1.117),
        ('R_n_HM', G[n][m]['R_n_HM'], 2.427),
        ('C_np_AM', G[n][m]['C_np_AM'],0.7425),
        ('C_rr_GM', G[n][m]['C_rr_GM'],2.155),
        ('c_rn_HM', G[n][m]['c_rn_HM'],195.71)
    ]
    for i in edge_results:
        print i

    return G

def check_gpickle(path):
    G= nx.read_gpickle(path)
    node = G.nodes_iter().next()
    (a,b) = G.edges_iter().next()
    nkey = set(G.node[node].keys())
    ekey = set(G[a][b].keys())
    for n in G.nodes():
        assert nkey == set(G.node[n].keys()), 'node not same: {}'.format(n)
        if nkey!=set(G.node[n].keys()):
            print 'node not same: {}'.format(n)
    for (a,b) in G.edges():
        assert ekey == set(G[a][b].keys()), 'edge not same: {}-{}'.format(a,b)
        if ekey!=set(G[a][b].keys()):
            print 'edge not same: {}-{}'.format(a,b)

    print 'Finish'
    return


if __name__=='main':
    G=nx.read_gpickle('data/undirected(fortest).gpickle')
    A = nx.adjacency_matrix(G,G.nodes(),weight='weight')
    A=A.toarray()
    np.fill_diagonal(A,0.1)
    Tri=np.diag(np.sum(A,axis=1))
    L=Tri-A
    Tri_1=np.diag(np.reciprocal(np.sqrt(Tri).diagonal()))
    Ls=Tri_1.dot(L).dot(Tri_1)
    #Ls, dd = graph_laplacian(A,normed=True, return_diag=True)
    sqrt_Tri=np.sqrt(Tri)

    c={}
    c[1]=np.array([1,0,0,0,0,0])
    c[2]=np.array([0,1,0,0,0,0])
    c[3]=np.array([0,0,0,1,0,0])
    c[4]=np.array([0,0,0,0,0,1])
    c[5]=np.array([0,0,1,0,1,0])

    sum=0

    for ar in c.values():
        u=sqrt_Tri.dot(ar)/np.linalg.norm(sqrt_Tri.dot(ar))
        sum=sum+u.dot(Ls).dot(u.T)

    print sum

