# import networkx as nx
# import matplotlib.pyplot as plt
#
# G1 = nx.Graph()
#
# G1.add_edges_from([('Frida','The Shawshank Redemption'), ('Pablo','The Shawshank Redemption'), ('Vincent','The Shawshank Redemption'), ('Joan', 'Forrest Gump'), ('Lee', 'Forrest Gump'), ('Andy', 'The Matrix'), ('Frida', 'The Matrix'), ('Pablo', 'The Matrix'),
# ('Andy', 'Anaconda'), ('Claude', 'Anaconda'), ('Georgia', 'Anaconda'), ('Frida', 'The Social Network'), ('Vincent', 'The Social Network'),
# ('Vincent', 'The Godfather'), ('Claude', 'Monty Python and the Holy Grail'), ('Georgia', 'Monty Python and the Holy Grail'), ('Claude', 'Snakes on a Plane'), ('Georgia', 'Snakes on a Plane'),
# ('Joan', 'Kung Fu Panda'), ('Lee', 'Kung Fu Panda'), ('Pablo', 'The Dark Knight'),
# ('Andy', 'Mean Girls'), ('Joan', 'Mean Girls'), ('Lee', 'Mean Girls')])
#
# l, r = nx.bipartite.sets(G1)
# pos = {}
#
# # Update position for node from each group
# pos.update((node, (1, index)) for index, node in enumerate(l))
# pos.update((node, (2, index)) for index, node in enumerate(r))
#
# print(len(G1.edges()))
#
# nx.draw(G1, pos=pos, with_labels=True)
# plt.show()


# import csv
# word_map = {}
# with open('../data2021/result_id2word.csv', 'r') as f:
#     reader = csv.reader(f)
#     for i in reader:
#         # print(i[0])
#         word_map[i[2]] = i[1]
# print(word_map)

# import pandas as pd
#
# obj_2 = pd.read_csv('../data2021/result_id2word.csv', index_col=[1])
# print(obj_2)
# import json
#
# with open('../data2021/wiki_word2id.json', encoding='utf8')as fp:
#     json_data = json.load(fp)
#     print('这是文件中的json数据：', json_data)
#     print('这是读取到文件数据的数据类型：', type(json_data))
#     print(len(json_data))  # 1279069

def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:", res)


print(next(foo()))


