import networkx  as nx
from tqdm import tqdm

G = nx.read_gpickle('data2021/test6_37_28_nominus.gpickle')

# for idx, (a, b, w) in tqdm(enumerate(G.edges(data='sem'))):
#     if G[a][b]['sem']<=1:
#         G[a][b]['oneminussem'] = 1-G[a][b]['sem']
#     if G[a][b]['sem']>1:
#         G[a][b]['oneminussem'] = 0
#
# G=nx.write_gpickle(G,'data2021/test6_37_28_nominus_updated.gpickle')


# data_sem = [4501088, 2158914, 957059, 6848165, 7149319, 36967, 1870794, 5679183, 1898453, 3529657, 4899866]
# data_explore_g_combined= [1511617, 4382280, 330697, 3942858, 4932428, 3031280, 2477716, 1525144, 4489531, 2786620, 4875197]
# data_explore_g_sta= [763744, 6218856, 1309035, 4932428, 3134015, 3761743, 3748272, 4936599, 4747199, 2418330, 22143]
# data_explore_s_combined= [2487232, 832418, 2235431, 3096074, 2260620, 4932428, 744845, 2239312, 1698320, 3348500, 3323900]
# data_explore_s_sta=[2235431, 747239, 3096074, 4001323, 2260620, 4932428, 2239312, 1698320, 5641072, 3348500, 1512510]

data_sem = [4501088, 2158914, 957059, 6848165, 7149319, 36967, 1870794, 5679183, 1898453, 3529657, 4899866]
data_path_g_combined= [601580, 140983, 4788728, 4932428]
data_path_g_sta= [601580, 6533884, 3761743, 22143,  6218856, 4932428]
data_path_s_combined= [601580, 2986080, 5679666, 5957976, 803145, 3885335, 4638157, 985919, 4962871, 5592112, 4932428]
data_path_s_sta=[601580, 1450027, 6586668, 4932428]

sum=0
for i in data_path_g_combined:
    sum=G.degree(i)+sum
    print ("data_path_g_combined")
print (sum)

sum=0
for i in data_path_g_sta:
    sum=G.degree(i)+sum
    print ("data_path_g_sta")
print (sum)
sum=0
for i in data_path_s_combined:
    sum=G.degree(i)+sum
    print ("data_path_s_combined")
print (sum)
sum=0
for i in data_path_s_sta:
    sum=G.degree(i)+sum
    print ("data_path_s_sta")
print (sum)


#
# data_search_path_s_combined =
# # 1.5041038310169825 [601580, 1246762, 6804391, 3582888]
# # 1.580152678176709 [601580, 140983, 3170451, 2012779, 3582888]
# # 1.5846171868059504 [601580, 660899, 6804391, 3582888]
# # 1.5864533648926922 [601580, 140983, 22143, 3170451, 2012779, 3582888]
# # 1.5963662117390418 [601580, 660899, 2602037, 3582888]
# # 1.5995000382593951 [601580, 6533884, 22143, 3170451, 2012779, 3582888]
#
#
# data_search_path_g_sta =
# data_search_path_s_combined =
#
# data_search_path_s_sta=