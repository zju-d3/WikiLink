import networkx as nx
import json
import csv
#
G=nx.read_gpickle('data2021/test6_37_28_nominus.gpickle')
with open('../webapp/data2021/word2id_from_graph.json', encoding='utf8')as fp:
    word_map = json.load(fp)
id2word=dict([(v,k) for (k,v) in word_map.items()])
# "\"3d printing\": 601580 "
# G.node[601580]

def get_weight(word):
    all_edges = G.edges(word_map[word])
    edges=dict()
    edges_combined = dict()
    for (a, b) in all_edges:
        edges_combined[id2word[b]]=G[a][b]['combined_sem_R_r_HM']
        edges[id2word[b]] = G[a][b]['R_r_HM']
    # edges_list[id2word[b]]=G[a][b]['combined_sem_R_n_HM']
    # edges_list[id2word[b]]=G[a][b]['combined_sem_R_n_GM']
    # edges_list[id2word[b]]=G[a][b]['combined_sem_R_r_GM']


    sort_edges=sorted(edges.items(),key=lambda item:item[1],reverse=False)
    sort_edges_combined=sorted(edges_combined.items(),key=lambda item:item[1],reverse=False)
# print(len(sort_edges))
# print(len(sort_edges_combined))

    result = dict()
    for i in range(len(sort_edges_combined)):
        result[sort_edges_combined[i][0]] = [sort_edges_combined[i][1], i+1]

    for i in range(len(sort_edges)):
        result[sort_edges[i][0]].append(sort_edges[i][1])
        result[sort_edges[i][0]].append(i+1)


    result=sorted(result.items(),key=lambda item:item[1][0],reverse=False)
# print(result)

    # print(result)
    final_result1 = []
    for i in result:
        item = [i[0]] + i[1]
        final_result1.append(item)

    result = sorted(result, key=lambda item: item[1][2], reverse=False)
    # print(result)

    # print(result)
    final_result2 = []
    for i in result:
        item = [i[0]] + i[1]
        final_result2.append(item)
    return final_result1, final_result2

first_step1, first_step2 = get_weight("user interface")

def get_next_step(parent, is_combined, stepnum):
    second_step1, second_step2 = get_weight(parent[0][0])

    if is_combined:
        for i in second_step1:
            i[1] += parent[0][1]
            i.append(stepnum)
        parent = parent + second_step1
        parent = sorted(parent, key=lambda item: item[1], reverse=False)
        return parent
    else:
        for i in second_step2:
            i[3] += parent[0][3]
            i.append(stepnum)
        parent = parent + second_step2
        parent = sorted(parent, key=lambda item: item[3], reverse=False)
        return parent
step2_for_one = get_next_step(first_step1, True, "step two")
fp = open("find_path_result/userinterface_two_1.csv", "w", encoding="utf8")
writer = csv.writer(fp)
writer.writerows(step2_for_one)
fp.close()

step2_for_two = get_next_step(first_step2, False, "step two")
fp = open("find_path_result/userinterface_two_2.csv", "w", encoding="utf8")
writer = csv.writer(fp)
writer.writerows(step2_for_two)
fp.close()
# print(final_result)


# a= [1103.4444444444446, 0, 0.12621311906527308, 4]
# b = 'brooklyn'
# a.insert(0, b)
# print(a)

# final_result = [i[1].insert(0, i[0]) for i in result]
# print(final_result)

# edges_combined=sorted(edges_combined.items(),key=lambda item:item[1],reverse=True)
# print("statistic&semantic")
# print(edges_combined)
#
# plt.barh([item[0] for item in edges], [item[1] for item in edges], align = 'center', height = 2)
# plt.legend()
# plt.figure(dpi=800)
# plt.xlabel('word')
# plt.ylabel('weight')
# plt.title(u'edges')
#
# plt.show()


# plt.barh([item[0] for item in edges_combined], [item[1] for item in edges_combined])
# plt.legend()
#
# plt.xlabel('weight')
# plt.ylabel('word')
# plt.title(u'edges_combined')
#
# plt.show()

# params

# x: 条形图x轴
# y：条形图的高度
# width：条形图的宽度 默认是0.8
# bottom：条形底部的y坐标值 默认是0
# align：center / edge 条形图是否以x轴坐标为中心点或者是以x轴坐标为边缘
