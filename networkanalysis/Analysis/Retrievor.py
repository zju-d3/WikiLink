# this module is for the existing relevant information retrieval.
import networkx as nx
import PF
import collections
# from sklearn.utils.graph import graph_laplacian
from scipy.sparse import csgraph
# from sklearn.utils.arpack import eigsh
from scipy.sparse.linalg import eigsh
import numpy as np
import math
from sklearn.cluster.k_means_ import k_means
import itertools
from heapq import heappush, heappop
from itertools import count
from user_Feedback import recordUser
from networkx.utils import open_file
import pickle
import csv


@open_file(0, mode="rb")
def read_GG(path):
    G = pickle.load(path, encoding='iso-8859-1')
    return G


class UndirectedG(object):
    # Read Graph, define schema
    def __init__(self, data_version, schema, userSchema):

    # new : read_csv
    # def __init__(self, data_version):
        # self.G = nx.read_gpickle('../{}.gpickle'.format(data_version))
        print('{}.gpickle'.format(data_version))
        self.G = read_GG('../{}.gpickle'.format(data_version))
        self.data_version = data_version
        print("----Read graph")
        self.schema = schema
        self.userSchema = userSchema
        assert type(self.G) == nx.classes.graph.Graph, 'Not undirected graph'
        print("----Connect mysql")
        self.user_generators = {}
        self.gencode = {'get_Rel_one': self.get_Rel_one, 'find_paths': self.get_pathsBetween_twonodes,
                        'find_paths_clusters': self.get_pathsBetween_twoClusters}

    # return ids list of input words
    # Inputs is a list of words
    def input_ids(self, ipts):
        # try:
        #    ipwids = PF.find_id(ipts, self.cursor)
        # except:
        cnx, cursor = PF.creatCursor(self.schema, 'R')

        ipwids = PF.find_id(ipts, cursor)
        ipwids = list(set(ipwids))
        for n in ipwids:
            # if condition returns False, AssertionError is raised:
            assert self.G.has_node(n), 'graph can not find the node'
            # print(n, self.G.node[n]['label'])

        cursor.close()
        cnx.close()
        return ipwids

    def input_ids_read_csv(self, word_map, ipts):
        ipwids, new_ipts = PF.find_id_csv(ipts, word_map)
        ipwids = list(set(ipwids))
        for n in ipwids:
            # print(type(n))
            # print(len(n))
            print(n, ": ", self.G.has_node(n))
            # if condition returns False, AssertionError is raised:
            assert self.G.has_node(n), 'graph can not find the node'
            # print(n, self.G.node[n]['label'])
        return ipwids, new_ipts

    # Use shortest path algorithms to get neighbors for one input
    # Get all the neighbors within l level
    # Get the neighbors of each level whihin l level
    # retured type is node
    # OK checked
    def get_Neib_one(self, ipt, l):
        nei = nx.single_source_shortest_path_length(self.G, ipt, cutoff=l)
        NB = {}
        for key, value in nei.iteritems():
            NB.setdefault(value, set()).add(key)
            NB.setdefault('AllNb', set()).add(key)

        return NB

    # Use shortest path algorithm to get neighbors
    # get neighbors of each input within l level
    # get neighbors of each level for each input
    # get unioned neighbors of all inputs within l level
    # returned type is node
    # OK checked
    def get_Neib(self, l, ipts):
        NB_Eachipt = {}
        NB_Allipts = set()
        for ipt in ipts:
            NB_Eachipt[ipt] = self.get_Neib_one(ipt, l)
            NB_Allipts.update(NB_Eachipt[ipt]['AllNb'])

        return {"NB_Eachipt": NB_Eachipt, "NB_Allipts": NB_Allipts}

    def get_Rel_one(self, ipt, tp, minhops, localnodes=None):
        print("get_Rel_one, 进来了！")
        """
        Generator of the most relevant words and their corresponding paths for an input.
        The word is sorted by the distance of its shortest path from input, and is filtered by minhops of its shortest path.
        The corresponding path of the word is its shortest path.

        :param ipt: source
        :param tp: the property of edge to be used as distance
        :param minhops: the minimum hops that the word's shortest path should contain
        :param localnodes: if none, find the relevent words in whole graph. Else, find the relevant words in localgraph.
        :return: (length, word's shortest path)
        """
        if localnodes == None:
            G = self.G
        else:
            G = self.G.subgraph(localnodes)

        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances
        seen = {ipt: 0}
        c = itertools.count()
        fringe = []  # use heapq with (distance,label) tuples
        push(fringe, (0, next(c), ipt))  # every element: (distance, index, nodeId)
        paths = {ipt: [ipt]}
        while fringe:
            # print("getRelOne: fringe: ", fringe)
            # print("getRelOne: paths: ", paths)
            (d, _, v) = pop(fringe)
            print("getRelOne: dist: ", dist)
            if v in dist:
                continue  # already searched this node

            dist[v] = d
            print("len(paths[v]): ", len(paths[v]))
            print("minhops: ", minhops)

            if len(paths[v]) >= minhops + 1:
                yield (d, paths[v])

            # print("getRelOne: G.adj[v].keys(): ", G.adj[v].keys())

            for u in G.adj[v].keys():
                cost = G[u][v][tp]  # {'weight': 10, 'maxAlpha': 0.994797281873221, 'minAlpha': 0.8355365180566858, 'R_r_GM': 1.690497337172318, 'R_r_HM': 6.4, 'R_n_GM': 9.866792191205915, 'R_n_HM': 19279.395061728395}
                # print("cost: ", cost)

                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if u in dist:
                    if vu_dist < dist[u]:
                        raise ValueError('Contradictory paths found:', 'negative weights?')
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    paths[u] = paths[v] + [u]

    def get_pathsBetween_twonodes(self, source, target, tp, minhops, localnodes=None):
        """
        Generator of shortest paths between two nodes using bidirectional_dijkstra, starting from shortest length.
        This method is faster than single_directional_dijkstra in large graph

        :param source: start node
        :param target: end node
        :param tp: edge property to be distance
        :param minhops: the minimum hops the generated path should contain
        :param localnodes: if none find the path in the whole graph, else find the path in the local graph.
        :return: (length, path)
        """

        if localnodes == None:
            G = self.G
        else:
            G = self.G.subgraph(localnodes)

        for (length, path) in shortest_simple_paths(G, source, target, weight=tp):
            if len(path) >= minhops + 1:
                yield (length, path)

    def my_Gen(self, N, user, parameters, generator, start=True):
        """
        Transform a generator to a function can go forward and backward

        :param N: Number of words/paths to be retrieved. If N is positive, Next N words. If N is negetive, previous N words.
        :param user: user's email
        :param parameters: parameters for the generator
        :param generator: the function of the generator
        :param start: If true, find the first N words/paths. Otherwise, find the next or previous N words/paths
        :return: nodes, paths. nodes is the union set of all nodes in the paths.
        """

        results = {}
        query_type = generator
        generator = self.gencode[generator]

        if start == True:
            self.user_generators[user] = {}
            self.user_generators[user]['generator'] = generator(**parameters)
            self.user_generators[user]['records'] = []
            self.user_generators[user]['Endpo'] = 0
            self.user_generators[user]['max'] = None

        self.user_generators[user]['Endpo'] += N
        startposition = self.user_generators[user]['Endpo'] - int(math.fabs(N))

        print("user_generatore:", self.user_generators[user])
        print("startposition: ", startposition)

        n_records = len(self.user_generators[user]['records'])
        if self.user_generators[user]['Endpo'] >= n_records:
            for i in range(self.user_generators[user]['Endpo'] - n_records + 1):
                try:
                    # length, path = self.user_generators[user]['generator'].next()
                    length, path = next(self.user_generators[user]['generator'])
                except:
                    self.user_generators[user]['max'] = len(self.user_generators[user]['records'])

                    # Record user error
                    if self.user_generators[user]['max'] == 0:
                        distance_type = parameters['tp']
                        start_id, start_label, end_id, end_label = recordUser.error_parameters(self.G, query_type,
                                                                                               parameters)
                        errthread = recordUser.error_thread(self.userSchema, self.data_version, distance_type, user,
                                                            query_type, start_id, start_label, end_id, end_label)
                        errthread.start()

                    break
                else:
                    self.user_generators[user]['records'].append(path)

        if startposition < 0:
            startposition += int(math.fabs(N))
            self.user_generators[user]['Endpo'] += int(math.fabs(N))
        if self.user_generators[user]['max'] is not None and startposition >= self.user_generators[user]['max']:
            startposition -= int(math.fabs(N))
            self.user_generators[user]['Endpo'] -= int(math.fabs(N))
        print("startposition: ", startposition)
        print("user_generatore:", self.user_generators[user])
        print("len(records): ", len(self.user_generators[user]['records']))

        results['allpaths'] = self.user_generators[user]['records'][startposition:self.user_generators[user]['Endpo']]
        results['allnodes'] = set()
        finalpaths = []
        for path in results['allpaths']:
            results['allnodes'].update(path)
            lapath = [self.G.node[n]['label'] for n in path]
            finalpaths.append({'ids': path, 'labels': lapath})

        print("results['allnodes']: ", results['allnodes'])
        print("finalpaths: ", finalpaths)
        print("len(finalpaths):", len(finalpaths))
        return results['allnodes'], finalpaths, startposition + 1

    def sort_clustersCentrality(self, clusters, distance):
        """
        This method sorts the nodes of each clusters based on their centrality in descending.
        The first node in each list has highest centrality.

        This method uses closeness centrality. But if possible, we can try other centrality such as betweenness, eigenvector,
        degree, or eccentricity, etc.

        :param clusters: array of lists. Each list contains the nodes of a cluster
        :param distance: distance is the edge attribute as the distance
        :return: array of lists. Each list is a cluster sorted by centrality.
        """

        def sort_oneCluster(cluster, distance):
            G = self.G.subgraph(cluster)
            centrality = [nx.closeness_centrality(G, u=n, distance=distance) for n in cluster]
            sortcluster = [y for (x, y) in sorted(zip(centrality, cluster), reverse=True)]
            return sortcluster

        clusters = [sort_oneCluster(cluster, distance) for cluster in clusters]

        return clusters

    def cutgraph(self, nodes, k, weight='weight', algorithm='normalized', Mx='LsLa'):
        """
        applying clustering on the subgraph consisting of the input nodes
        This method may be faster than np.linalg.eigh( )

        Paramters
        ----------
        nodes: a list of node to be clustered. The subgraph projected by the nodes can be connected or unconnected.

        k: the number of clusters to be generated

        weight: is the kernal value between two nodes. Higher kernal value means the two nodes are more similar and closer.
                Here the weight can be the original frequence that two words appear together.

        algorithm: {'normalized', 'modularity'}, default to 'normalized'
                   The performance of 'modularity' is NOT good.

        Mx: the matrix whose eigenvectors are used for the k-mean algorithm.

        Return
        ----------
        clusters: array of lists. Each list contains the nodes of a cluster
        """
        G = self.G.subgraph(nodes)
        # assert nx.is_connected(G)==True, "graph is not connected"
        A = nx.adjacency_matrix(G, weight=weight)

        if algorithm == "normalized":
            # Ls, dd = graph_laplacian(A, normed=True, return_diag=True)
            Ls, dd = csgraph.laplacian(A, normed=True, return_diag=True)

            eigenvalue_n, eigenvector_n = eigsh(Ls * (-1), k=k,
                                                sigma=1.0, which='LM',
                                                tol=0.0)

            if Mx == 'La':
                Trisq = np.sqrt(np.array(A.sum(axis=1)).transpose()[0])
                for ti, t in enumerate(Trisq):
                    if t == 0:
                        Trisq[ti] = 1.0
                for j in range(0, eigenvector_n.shape[1]):
                    eigenvector_n[:, j] = eigenvector_n[:, j] / Trisq
                    colnorm = np.linalg.norm(eigenvector_n[:, j])
                    eigenvector_n[:, j] = eigenvector_n[:, j] / colnorm

            elif Mx == 'LsLa':
                # eigenvector for eigenvalue zero
                components = nx.connected_components(G)
                i_comp = 0
                while True:
                    try:
                        component = components.next()
                        i_comp += 1
                    except:
                        break
                    else:
                        if i_comp > k:
                            break
                        else:
                            sq_comp = 1.0 / math.sqrt(len(component))
                            vec_comp = []
                            for n in G.nodes():
                                if n in component:
                                    vec_comp.append(sq_comp)
                                else:
                                    vec_comp.append(0.0)
                            eigenvector_n[:, -i_comp] = np.array(vec_comp)

            else:
                assert Mx == 'Ls', 'type of matrix is not known'



        elif algorithm == "modularity":
            tr = np.sum(A)
            d = np.sum(A, axis=1)
            Q = (A - (d * d.T) / tr) / tr
            eigenvalue_n, eigenvector_n = eigsh(Q, k=k,
                                                sigma=1.0, which='LM',
                                                tol=0.0)
            for i, vl in enumerate(eigenvalue_n):
                if vl > 1e-10:
                    eigenvector_n = eigenvector_n[:, i:]
                    break

        else:
            raise (TypeError, "unrecognized algorithm: {}".format(algorithm))

        # normalize row vector
        for i, v in enumerate(eigenvector_n):
            rownorm = float(np.linalg.norm(v))
            if rownorm != 0:
                eigenvector_n[i] = v / rownorm

        _, labels, _ = k_means(eigenvector_n, k, random_state=None,
                               n_init=10)

        dic_clusters = {}
        for index, n in enumerate(G.nodes()):
            dic_clusters.setdefault(labels[index], list()).append(n)

        clusters = dic_clusters.values()

        return clusters

    def mcl_cluster(self, nodes, r, weight='weight'):
        """
        Applying Markov clustering on the input nodes

        :param nodes: a list of node to be clustered.

        :param r: inflation factor

        :param weight: is the kernal value between two nodes. Higher kernal value means the two nodes are more similar and closer.
                Here the weight can be the original frequence that two words appear together.

        :return M: the convergent matrix

        :return clusters: array of lists. Each list contains the nodes of a cluster
        """

        def normalize_matrix(adjacency):
            Tri = adjacency.sum(axis=1)
            M = adjacency / Tri
            return M

        def get_cluster(mx, nodes):
            queue = set(nodes)
            diag = np.array(mx).diagonal()
            clusters = {}
            for i, d in enumerate(diag):
                if d >= 1e-5:
                    if nodes[i] in queue:
                        queue.remove(nodes[i])
                        clusters.setdefault(nodes[i], set()).add(nodes[i])
                        for j, jd in enumerate(np.array(mx)[:, i]):
                            if jd >= 1e-5 and j != i:
                                clusters[nodes[i]].add(nodes[j])
                                queue.remove(nodes[j])
                    else:
                        continue

            if len(queue) != 0:
                raise (TypeError, "mcl_cluster miss nodes")
            for s1, s2 in itertools.combinations(clusters.keys(), 2):
                if clusters[s1] & clusters[s2]:
                    raise (TypeError, "mcl_cluster overlapping cluster")

            clusters = [list(cl) for cl in clusters.values()]
            return clusters

        G = self.G.subgraph(nodes)
        A = nx.adjacency_matrix(G, weight=weight).todense()

        # use np.fill_diagonal(A, 2*np.sum(A, axis=1) + 1.0) if you want to quickly divde smaller clusters
        np.fill_diagonal(A, np.sum(A, axis=1) + 1.0)

        M = normalize_matrix(A)

        while True:
            nM = np.linalg.matrix_power(M, 2)
            nM = np.power(nM, r)
            nM = normalize_matrix(nM)
            er = np.linalg.norm(nM - M)
            M = nM
            if er <= 1e-5:
                break

        clusters = get_cluster(M, G.nodes())

        return M, clusters

    def generate_Bpaths(self, cluster1, cluster2, tp):
        """
        Generator of B-paths between cluster1 and cluster2, ordered by length from shortest to longest
        Note: this method uses single directional dijkstra algorithm which will be very slow in large graph.

        :param cluster1: list of nodes in cluster1

        :param cluster2: list of nodes in cluster2

        :param tp: the property of edge to be considered as distance

        :return: generate (length,B-path)
        """
        cset1 = set(cluster1)
        cset2 = set(cluster2)
        push = heappush
        pop = heappop

        fringe = []  # queue to sort the B-paths

        for n in cluster1:
            push(fringe, (0, [n]))

        while fringe:
            (d, p) = pop(fringe)
            end = p[-1]

            if end in cset2:  # reach cluster2
                yield (d, p)
                continue

            for nei in self.G.adj[end].keys():  # search near end node
                if nei not in cset1 and nei not in p:
                    up_d = d + self.G[nei][end][tp]
                    up_p = p + [nei]
                    push(fringe, (up_d, up_p))

    def get_pathsBetween_twoClusters(self, cluster1, cluster2, tp, localnodes=None):
        """
        generator of shortest paths between two clusters using bidirectional_dijkstra. Starting from shortest.
        This method is faster than single directional dijkstra in large graph

        :param cluster1: list of nodes in cluster1
        :param cluster2: list of nodes in cluster2
        :param tp: edges property to be the distance
        :param localnodes: if none find the paths in whole graph, else find the paths in the local graph
        :return: (length, path)
        """

        if localnodes == None:
            G = self.G
        else:
            G = self.G.subgraph(localnodes)

        for (length, path) in shortest_simple_pathsForClusters(G, cluster1, cluster2, weight=tp):
            yield (length - 0.2, path[1:-1])


# -----------------------functions-----------------------------------------
def shortest_simple_paths(G, source, target, weight=None):
    if source not in G:
        raise nx.NetworkXError('source node %s not in graph' % source)

    if target not in G:
        raise nx.NetworkXError('target node %s not in graph' % target)

    if weight is None:
        raise TypeError('weight not defined')
    else:
        def length_func(path):
            return sum(G.edge[u][v][weight] for (u, v) in zip(path, path[1:]))

        shortest_path_func = bidirectional_dijkstra

    listA = list()
    listB = PathBuffer()
    prev_path = None
    while True:
        if not prev_path:
            length, path = shortest_path_func(G, source, target, weight=weight)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                ignore_nodes.add(root[-1])
                try:
                    length, spur = shortest_path_func(G, root[-1], target,
                                                      ignore_nodes=ignore_nodes,
                                                      ignore_edges=ignore_edges,
                                                      weight=weight)
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass

        if listB:
            length_of_path, path = listB.pop()
            yield (length_of_path, path)
            listA.append(path)
            prev_path = path
        else:
            break


def bidirectional_dijkstra(G, source, target, weight='weight',
                           ignore_nodes=None, ignore_edges=None):
    if ignore_nodes:
        ignore_nodes = set(ignore_nodes)
        if source in ignore_nodes:
            ignore_nodes.remove(source)
        if target in ignore_nodes:
            ignore_nodes.remove(target)

    if source == target:
        return (0, [source])

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors_iter
        Gsucc = G.successors_iter
    else:
        Gpred = G.neighbors_iter
        Gsucc = G.neighbors_iter

    # support optional nodes filter
    if ignore_nodes:
        def filter_iter(nodes_iter):
            def iterate(v):
                for w in nodes_iter(v):
                    if w not in ignore_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    # support optional edges filter
    if ignore_edges:
        if G.is_directed():
            def filter_pred_iter(pred_iter):
                def iterate(v):
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w

                return iterate

            def filter_succ_iter(succ_iter):
                def iterate(v):
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)

        else:
            def filter_iter(nodes_iter):
                def iterate(v):
                    for w in nodes_iter(v):
                        if (v, w) not in ignore_edges \
                                and (w, v) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)

    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [{source: [source]}, {target: [target]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{source: 0}, {target: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if (dir == 0):  # forward
                if G.is_multigraph():
                    minweight = min((dd.get(weight, 1)
                                     for k, dd in G[v][w].items()))
                else:
                    minweight = G[v][w].get(weight, 1)
                vwLength = dists[dir][v] + minweight  # G[v][w].get(weight,1)
            else:  # back, must remember to change v,w->w,v
                if G.is_multigraph():
                    minweight = min((dd.get(weight, 1)
                                     for k, dd in G[w][v].items()))
                else:
                    minweight = G[w][v].get(weight, 1)
                vwLength = dists[dir][v] + minweight  # G[w][v].get(weight,1)

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError(
                        "Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))


def shortest_simple_pathsForClusters(G, cluster1, cluster2, weight=None):
    cset1 = set(cluster1)
    cset2 = set(cluster2)
    if not cset1.issubset(set(G.nodes())):
        raise nx.NetworkXError('cluster1 nodes not in graph')

    if not cset2.issubset(set(G.nodes())):
        raise nx.NetworkXError('cluster2 nodes not in graph')

    if weight is None:
        raise TypeError('weight not defined')
    else:
        def length_func(path):
            length = 0
            for (u, v) in zip(path, path[1:]):
                if set([u, v]) & set(['source', 'target']):
                    length += 0.1
                else:
                    length += G.edge[u][v][weight]
            return length

        shortest_path_func = bidirectional_dijkstra_forClusters

    listA = list()
    listB = PathBuffer()
    prev_path = None
    ori_ignEdges = set(G.subgraph(cluster1).edges() + G.subgraph(cluster2).edges())
    while True:
        if not prev_path:
            length, path = shortest_path_func(G, 'source', 'target', cluster1, cluster2, ignore_edges=set(ori_ignEdges),
                                              weight=weight)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set(ori_ignEdges)
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                ignore_nodes.add(root[-1])
                try:
                    length, spur = shortest_path_func(G, root[-1], 'target', cluster1, cluster2,
                                                      ignore_nodes=ignore_nodes,
                                                      ignore_edges=ignore_edges,
                                                      weight=weight)
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass

        if listB:
            length_of_path, path = listB.pop()
            yield (length_of_path, path)
            listA.append(path)
            prev_path = path
        else:
            break


class PathBuffer(object):

    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return cost, path


#####
def bidirectional_dijkstra_forClusters(G, source, target, cluster1, cluster2, weight='weight',
                                       ignore_nodes=set(), ignore_edges=None):
    """Dijkstra's algorithm for shortest paths using bidirectional search.

    This function returns the shortest path between source and target
    ignoring nodes and edges in the containers ignore_nodes and
    ignore_edges.

    This is a custom modification of the standard Dijkstra bidirectional
    shortest path implementation at networkx.algorithms.weighted

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node.

    target : node
       Ending node.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    Returns
    -------
    length : number
        Shortest path length.

    Returns a tuple of two dictionaries keyed by node.
    The first dictionary stores distance from the source.
    The second stores the path from the source to that node.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    In practice  bidirectional Dijkstra is much more than twice as fast as
    ordinary Dijkstra.

    Ordinary Dijkstra expands nodes in a sphere-like manner from the
    source. The radius of this sphere will eventually be the length
    of the shortest path. Bidirectional Dijkstra will expand nodes
    from both the source and the target, making two spheres of half
    this radius. Volume of the first sphere is pi*r*r while the
    others are 2*pi*r/2*r/2, making up half the volume.

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    shortest_path
    shortest_path_length
    """
    if ignore_nodes:
        ignore_nodes = set(ignore_nodes)
        if source in ignore_nodes:
            ignore_nodes.remove(source)
        if target in ignore_nodes:
            ignore_nodes.remove(target)

    if source == target:
        return (0, [source])

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors_iter
        Gsucc = G.successors_iter
    else:
        Gpred = G.neighbors_iter
        Gsucc = G.neighbors_iter

    # support optional nodes filter
    # if ignore_nodes:
    def filter_nodes(nodes_iter):
        def iterate(v):
            if v == 'source':
                nb = cluster1
            elif v == 'target':
                nb = cluster2
            else:
                nb = list(nodes_iter(v))
                if v in cluster1:
                    nb += ['source']
                if v in cluster2:
                    nb += ['target']

            for w in nb:
                if w not in ignore_nodes:
                    yield w

        return iterate

    Gpred = filter_nodes(Gpred)
    Gsucc = filter_nodes(Gsucc)

    # support optional edges filter
    if ignore_edges:
        if G.is_directed():
            def filter_pred_iter(pred_iter):
                def iterate(v):
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w

                return iterate

            def filter_succ_iter(succ_iter):
                def iterate(v):
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)

        else:
            def filter_edges(nodes_iter):
                def iterate(v):
                    for w in nodes_iter(v):
                        if (v, w) not in ignore_edges \
                                and (w, v) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_edges(Gpred)
            Gsucc = filter_edges(Gsucc)

    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [{source: [source]}, {target: [target]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{source: 0}, {target: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if (dir == 0):  # forward
                if v != 'source' and w in cluster1:
                    continue
                if v in cluster2 and w != 'target':
                    continue
                if G.is_multigraph():
                    minweight = min((dd.get(weight, 1)
                                     for k, dd in G[v][w].items()))
                else:
                    if set([w, v]) & set(['source', 'target']):
                        minweight = 0.1
                    else:
                        minweight = G[v][w].get(weight, 1)
                vwLength = dists[dir][v] + minweight  # G[v][w].get(weight,1)
            else:  # back, must remember to change v,w->w,v
                if v != 'target' and w in cluster2:
                    continue
                if v in cluster1 and w != 'source':
                    continue
                if G.is_multigraph():
                    minweight = min((dd.get(weight, 1)
                                     for k, dd in G[w][v].items()))
                else:
                    if set([w, v]) & set(['source', 'target']):
                        minweight = 0.1
                    else:
                        minweight = G[w][v].get(weight, 1)
                vwLength = dists[dir][v] + minweight  # G[w][v].get(weight,1)

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError(
                        "Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))
