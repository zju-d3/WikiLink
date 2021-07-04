import pickle
import networkx as nx
from networkx.utils import open_file
import matplotlib.pyplot as plt


@open_file(0, mode="rb")
def read_GG(path):
    G = pickle.load(path, encoding='iso-8859-1')
    return G


graph = read_GG("..//data2021//wiki_result_label_added_1_1.gpickle")
# graph = read_GG("..//addNodeEdgeDegree_G+SP+R_undirected_alpha0.65_nodeD1.0_total_v3_csvneo4j.gpickle")

# print(len(graph.nodes()))  # 218002

print(graph.nodes())
# print(type(graph.nodes()))
# # print(graph.nodes()[0])
# print(graph.node[0]['n'])
# print(graph.node[0]['N'])
# print(graph.has_node(1186629))
# print(graph.node[1021707]['label'])  # t-closeness|''t''-closeness
# print(graph.node[496456]['label'])  # supertype
# print(graph.node[29434]['label'])  # fork (software development)|fork
# print(graph.node[0]['label'])
# print(graph.adj[496456].keys())
# print(graph.node[898225]['label'])  #

print(graph[496456][898225])



