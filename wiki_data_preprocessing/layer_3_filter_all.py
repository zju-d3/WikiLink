

import os
from tqdm import tqdm
import pandas as pd
import networkx as nx
import json
import csv
import re
import numpy as np
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt


def read_technology_categories(directory):
    """
    to read the technology categories crawled from wiki
    :param directory:  the path of technology category, example file is 'categories_list_3.csv'
    :return:  a list of cates
    """
    df = pd.read_csv(directory)
    cate = df['#category#']
    cate = cate.tolist()
    return (cate)




def list_full_paths(directory):
    """
    to have the full absolute paths of the wiki raw data
    :param directory: relative path
    :return: absolute path
    """
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def read_technology_related_items(csv_file_directory,valid_cate):
    """
    循环读取wiki dump里的内容(已经筛选过，只留去hyperlinks和see also），通过判断wiki item下的category是不是在technology cate里 决定是否取这个wiki item
    :param csv_file_directory:  存这个item的地址
    :return: 没有return
    """

    files = list_full_paths('../wiki')

    with open(csv_file_directory, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', )
        csv_writer.writerow(['#item#'])
        for file in tqdm(files):
            print(file)
            if not file.endswith('.ipynb_checkpoints') and not file.endswith('.ipynb'):
                with open(file, 'r') as f:
                    doc = json.load(f)
                    cates = doc['cate_holder']
                    i = 0
                    for key, values in cates.items():
                        if values is not None:
                            for value in values:
                                if value in valid_cate:
                                    csv_writer.writerow([key])
                                    break
### shouldn't have lower.() here ###
### deleted .lower() here ## modify afterwards##


def filtered_meaning_less(csv_file_directory, filtered_csv_file_directory):
    """
    正则表达去除无意义的item
    :param csv_file_directory:
    :return: 无return， 直接写入文件
    """
    df = pd.read_csv(csv_file_directory)
    for i, item in tqdm(enumerate(df['#item#'])):
        if type(item) is not str:
            df.drop(labels=i, axis=0, inplace=True)
        else:
            boole = re.match('.+:.+', item)

            if boole is not None:
                #                 print (boole)
                #                 print (item)
                df.drop(labels=i, axis=0, inplace=True)
        if i % 10000 == 0:
            print(i)
    df.to_csv(filtered_csv_file_directory, index=False)

    return (df['#item#'].tolist())

def relate_holder_filter_not_sequential(csv_file_directory, relate_holder_diretory):
    """
    根据得到的technology下的item, 然后筛选item相关的 relates/hyperlinks
    this is the way in a sequenntial extracting because csv file and wiki are in the same way
    :param csv_file_directory:technology 下的item
    :param relate_holder_diretory:存放relate的地方
    :return: 无return， 直接写入文件
    """
    df = pd.read_csv(csv_file_directory,names = ['#item#'])
    targets = list(df['#item#'])
    new_holder = dict()
    length_reminder = len(targets)
    files = list_full_paths('../wiki')
    i = 0
    for file in tqdm(files):
        print(file)
        if not file.endswith('.ipynb_checkpoints') and not file.endswith('.ipynb'):
            with open(file, 'r') as f:
                doc = json.load(f)
                #     cates = doc['cate_holder']
                relates = doc['relate_holder']
                # print (relates)
                #     see_also=doc['see_also_holder']

                for key in relates.keys():
                    # target = targets[i]

                    if key in targets:
                        targets.remove(key)
                        new_holder[key] = relates[key]
                        i = i + 1
                        if i == length_reminder:
                            break




    with open(relate_holder_diretory, 'w') as f:
        json.dump(new_holder, f)


def see_also_holder_filter_not_sequential(csv_file_directory, see_also_holder_diretory):
    """
    根据得到的technology下的item, 然后筛选item相关的 see also

    :param csv_file_directory: technology 下的item
    :param see_also_holder_diretory:  存放relate的地方
    :return: 无return 直接写入
    """
    df = pd.read_csv(csv_file_directory, names=['#item#'])
    targets = list(df['#item#'])
    length_reminder = len(targets)
    new_see_also_holder = dict()
    # print (targets [0:1000])
    i = 0
    ### i=0 put here for consequential indexing ####
    files = list_full_paths('../wiki')

    for file in tqdm(files):
        if not file.endswith('.ipynb_checkpoints') and not file.endswith('.ipynb'):
            with open(file, 'r') as f:
                doc = json.load(f)
                #     cates = doc['cate_holder']
                #             relates= doc['relate_holder']
                see_also = doc['see_also_holder']

                for key in see_also.keys():
                    # target = targets[i]

                    if key in targets:
                        targets.remove(key)
                        new_see_also_holder[key] = see_also[key]
                        i = i + 1
                        if i == length_reminder:
                            break



    with open(see_also_holder_diretory, 'w', ) as f:
        json.dump(new_see_also_holder, f)


def get_keywords_list(items_file_directory, relates_file_directory, see_also_file_directory, keyword_file_directory):
    """
    得到整个出现在item, hyperlinks, see also中的全部词 的 corpus， 注意！有重复，未去重
    :param items_file_directory: wiki_item， csv格式
    :param relates_file_directory: relates hyperlinks , json格式
    :param see_also_file_directory:  see also, json格式
    :param keyword_file_directory:  存放keyword的地址， csv格式
    :return: 无返回， 已写入
    """
    df = pd.read_csv(items_file_directory)
    keywords_list = list(df['#item#'])
    keyword_mapping = dict()

    with open(keyword_file_directory, 'w',encoding = 'utf-8',newline='') as cw:
        writer = csv.writer(cw)
        for keyword in keywords_list:
            # if (type(keyword)) =='str':
            if type(keyword) is not str:
                keyword = str(keyword)
            writer.writerow([keyword.lower()])

        i = 0
        with open(relates_file_directory, 'r',encoding ='utf-8') as f1:

            relates = json.load(f1)
            for key, values in relates.items():
                i = i + 1
                if i % 5000 == 0:
                    print(i)
                if values is not None:
                    for value in values:
                        writer.writerow([value.lower()])

        i = 0
        with open(see_also_file_directory, 'r',encoding ='utf-8') as f2:
            see_alsos = json.load(f2)
            for key, values in see_alsos.items():
                i = i + 1
                if i % 5000 == 0:
                    print(i)
                if values is not None:
                    for value in values:
                        writer.writerow([value.lower()])


def set_keywords_list(keyword_file_directory, set_keywords_list_directory):
    """
    去重corpus
    :param keyword_file_directory:   corpus原地址， csv格式
    :param set_keywords_list_directory:  corpus 新地址， csv格式
    :return:  无返回 ， 已写入
    """
    df1 = pd.read_csv(keyword_file_directory, names=['item'])
    keywords_list_set = list(df1['item'])
    keywords_list_set = set(keywords_list_set)
    with open(set_keywords_list_directory, 'w',encoding = 'utf-8') as f:
        writer = csv.writer(f)
        for keyword in keywords_list_set:
            writer.writerow([keyword])


def filter_set_keywords_list(set_keywords_list_directory, keywords_list_set_filtered_directory):
    """
    再筛
    :param set_keywords_list_directory:  去重后地址 ，格式csv
    :param keywords_list_set_filtered_directory:  去重后再筛的地址，格式csv
    :return:无返回， 已写入
    """
    df = pd.read_csv(set_keywords_list_directory)
    with open(keywords_list_set_filtered_directory, 'w',encoding = 'utf-8') as cw:
        writer = csv.writer(cw)
        for i, item in enumerate(df['nan']):
            try:
                boole = re.match('.+:.+', item)
                if boole is None:
                    #                     print (boole)
                    #                     print (item)
                    writer.writerow([df['nan'][i]])
                else:
                    print(item)
                if i % 10000 == 0:
                    print(i)
            except:
                raise TypeError('find :{}-{}'.format(item, type(item)))


def word2id_and_id2word_list(keywords_list_set_filtered_directory,word2id_directory,id2word_directory):
    """
    得到i2w, w2i 的list
    :param keywords_list_set_filtered_directory:  去重后再筛选的keyword地址， 格式csv
    :param word2id:  word2id， 格式json
    :param id2word:  id2word， 格式json
    :return:
    """
    df = pd.read_csv(keywords_list_set_filtered_directory, names=['item'])
    index = range(1, len(df) + 1)
    word2id = dict(zip(list(df['item']), index))
    id2word = dict(zip(index, list(df['item'])))
    word2id['UNK'] = 0
    id2word[0] = ['UNK']
    with open(word2id_directory, 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False)
    with open(id2word_directory, 'w', encoding='utf-8') as f:
        json.dump(id2word, f, ensure_ascii=False)


def word2id(word2id,filtered_relate_holder, filtered_see_also_holder,relate_holder_id,see_also_holder_id):
    """

    :param word2id:  word2id 地址
    :param filtered_relate_holder: relate地址 ， json格式
    :param filtered_see_also_holder: see also 地址， json格式
    :param relate_holder_id:  relate转成id形式地址， json格式
    :param see_also_holder_id:  see also转成id形式地址， json格式
    :return:无返回 已写入
    """


    with open(word2id, 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    with open(filtered_relate_holder, 'r') as f:
        relates = json.load(f)
    with open(filtered_see_also_holder, 'r') as f:
        see_alsos = json.load(f)
        id_relate_holder = dict()
    i = 0
    for key, values in relates.items():
        if type(key) is not str:
            key = str(key)

        id_relate_holder[word2id[key.lower()]] = []

        for value in values:
            if value.lower() in word2id.keys():

                id_relate_holder[word2id[key.lower()]].append(word2id[value.lower()])

            else:
                id_relate_holder[word2id[key.lower()]].append(word2id['UNK'])

    with open(relate_holder_id, 'w', encoding='utf-8') as f:
        json.dump(id_relate_holder, f, ensure_ascii=False)

    id_see_also_holder = dict()
    i = 0
    for key, values in see_alsos.items():
        if type(key) is not str:
            key = str(key)
        id_see_also_holder[word2id[key.lower()]] = []

        for value in values:
            if value.lower() in word2id.keys():

                id_see_also_holder[word2id[key.lower()]].append(word2id[value.lower()])

            else:
                id_see_also_holder[word2id[key.lower()]].append(word2id['UNK'])

    with open(see_also_holder_id, 'w', encoding='utf-8') as f:
        json.dump(id_see_also_holder, f, ensure_ascii=False)


def adding_item_to_relate_and_see_also(relates_id_path,see_also_id_pth,relates_id_added_path, see_also_id_added_path):
    """
    0:[1,2,3,4,5] --> 0:[1,2,3,4,5,0] , for the following process, it's convnient to do w2w mapping only in dictionary's
    value, no need to read key.
    :param relates_id_path:  original  relate_id path , in json format
    :param see_also_id_pth:  orignal see_also_path, in json format
    :param relates_id_added_path:   added relate_id path, in json format
    :param see_also_id_added_path:  added see_also_path, in json format
    :return: no return, result have written in file.
    """
    with open(relates_id_path, 'r', encoding='utf-8') as f:
        relates = json.load(f)


    relates_holder_id_added = dict()

    for key, values in relates.items():
        if len(values) == 0:
            relates_holder_id_added[key] = []
        else:
            values.append(int(key))
            relates_holder_id_added[key] = values
    with open(relates_id_added_path, 'w', encoding='utf-8') as f:
        json.dump(relates_holder_id_added, f)

    with open(see_also_id_pth, 'r', encoding='utf-8') as f:
        relates = json.load(f)


    relates_holder_id_added = dict()

    for key, values in relates.items():
        if len(values) == 0:
            relates_holder_id_added[key] = []
        else:
            values.append(int(key))
            relates_holder_id_added[key] = values
    with open(see_also_id_added_path, 'w', encoding='utf-8') as f:
        json.dump(relates_holder_id_added, f)


def w2w_csv_every(relate_holder_id_added,see_also_holder_id_added,w2w_csv):
    """
    This is the case that the relate holder and see_also holder is within the capacity of the local computer,
    which means, w2w weight value can be updated in this one function.


    0:[0,1,2,3,4,5]  --> (0,1,X(weight value) (0,2,X),(0,3,X),(0,4,X),(0,5,X),(1,2,X).....
    for everytime happens in relate , X add 1
    for everytime happens in see_also, X add 9


    :param relate_holder_id_added: the origanl added relate file path, in json format
    :param see_also_holder_id_added: the orignal added see_also file path, in json format
    :param w2w_csv:  w2w file path to write , in csv format
    :return: no return, result have written in file.
    """
    with open(relate_holder_id_added, 'r', encoding='utf-8') as f:
        relates = json.load(f)
        pairs = dict()
        count = 0
        print(len(relates))
        for value in relates.values():

            x = len(value)

            if x > 0:
                for i in range(0, x):
                    for j in range(i, x):
                        if value[i] >= value[j]:
                            if (value[j], value[i]) in pairs.keys():
                                pairs[(value[j], value[i])] = pairs[(value[j], value[i])] + 1
                            else:
                                pairs[(value[j], value[i])] = 1
                        if value[i] < value[j]:
                            if (value[i], value[j]) in pairs.keys():
                                pairs[(value[i], value[j])] = pairs[(value[i], value[j])] + 1
                            else:
                                pairs[(value[i], value[j])] = 1

            count = count + 1
            if count > 10000 and count % 10000 == 0:
                print(count)

    with open(see_also_holder_id_added, 'r', encoding='utf-8') as f:
        relates = json.load(f)

        count = 0
    for value in relates.values():

        x = len(value)

        if x > 0:
            for i in range(0, x):
                for j in range(i, x):
                    if value[i] >= value[j]:
                        if (value[j], value[i]) in pairs.keys():
                            pairs[(value[j], value[i])] = pairs[(value[j], value[i])] + 9
                        else:
                            pairs[(value[j], value[i])] = 1
                    if value[i] < value[j]:
                        if (value[i], value[j]) in pairs.keys():
                            pairs[(value[i], value[j])] = pairs[(value[i], value[j])] + 9
                        else:
                            pairs[(value[i], value[j])] = 1

        count = count + 1
        if count > 10000 and count % 10000 == 0:
            print(count)

    with open(w2w_csv, 'w', newline='') as f:
        csvwriter = csv.writer(f, delimiter=';')
        count = 0
        for key, value in pairs.items():
            count = count + 1

            csvwriter.writerow([key[0], key[1], value])

def w2w_csv_too_big(relate_holder_id_added,see_also_holder_id_added,w2w_csv):

    """
    This is the case that the relate holder and see_also holder is BEYOND the capacity of the local computer,
    which means, w2w weight value can't be updated in this one function.


    0:[0,1,2,3,4,5]  --> (0,1,X(weight value) (0,2,X),(0,3,X),(0,4,X),(0,5,X),(1,2,X).....
    for everytime happens in relate , X will be 1, if happens N times,  N lines will be write,instead of update the value
    for everytime happens in see_also, X will be 9, if happens N times,  N lines will be write,instead of update the value


    :param relate_holder_id_added: the origanl added relate file path, in json format
    :param see_also_holder_id_added: the orignal added see_also file path, in json format
    :param w2w_csv:  w2w file path to write , in csv format
    :return: no return, result have written in file.
    """

    with open(w2w_csv, 'w', newline='') as f:
        csvwriter = csv.writer(f, delimiter=';')

        with open(relate_holder_id_added, 'r', encoding='utf-8') as f:
            relates = json.load(f)
            count = 0
            print ('total:{}'.format(len(relates)))
            for value in relates.values():

                x = len(value)

                if x > 0:
                    for i in range(0, x):
                        for j in range(i, x):
                            if value[i] >= value[j]:
                                csvwriter.writerow([value[j],value[i],1])

                            if value[i] < value[j]:
                                csvwriter.writerow([value[i],value[j],1])

                count = count + 1
                if count > 10000 and count % 10000 == 0:
                    print(count)



        with open(see_also_holder_id_added, 'r', encoding='utf-8') as f:
            relates = json.load(f)

            count = 0
        for value in relates.values():

            x = len(value)

            if x > 0:
                for i in range(0, x):
                    for j in range(i, x):
                        if value[i] >= value[j]:
                            csvwriter.writerow([value[j], value[i], 1])

                        if value[i] < value[j]:
                            csvwriter.writerow([value[i], value[j], 1])

            count = count + 1
            if count > 10000 and count % 10000 == 0:
                print(count)

def get_node_degree_set(word2id_path,relate_holder_id_added,see_also_holder_id_added,node_degree_set_json):
    """
    for the very big relate id iadded file, filter it based on the node degree first for further processing.
    0：[1,2,3,4,5,5,5,5]  1:[2,6,7,0]
    remove the dupliate node in same relate holder, also the dupicate node globally
    e.g. Node degre  of "0" is : 5 + 2 = 7

    :param word2id_path:  the word2id file path, in json format, used to have a dictionary's value to store the node degree
    :param relate_holder_id_added:  relate id added file path, in json format
    :param see_also_holder_id_added:  see also id added file path,  in json format
    :param node_degree_json:
    :return:
    """
    with open (word2id_path,'r',encoding='utf-8') as f:
        dic=json.load(f)
    length= len(dic)
    node_count_dic=dict()
    for i in range(length):
        node_count_dic[i]=[]
    with open(relate_holder_id_added, 'r', encoding='utf-8') as f:
        relates = json.load(f)
        count = 0
        print('total:{}'.format(len(relates)))
        for value in tqdm(relates.values()):

            x = len(value)

            if x > 0:
                for i in range(0, x):
                    if value[i]!=0:
                        node_count_dic[value[i]]=list(set(node_count_dic[value[i]]+list(value)))

            count = count + 1
            if count > 10000 and count % 10000 == 0:
                print(count)

    with open(see_also_holder_id_added, 'r', encoding='utf-8') as f:
        see_alsos= json.load(f)
        count = 0
        for value in see_alsos.values():

            x = len(value)

            if x > 0:

                for i in range(0, x):
                    if value[i] != 0:
                        node_count_dic[value[i]]=list(set(node_count_dic[value[i]]+list(value)))

            count = count + 1
            if count > 10000 and count % 10000 == 0:
                print(count)
    with open(node_degree_set_json,'w',encoding='utf-8') as f:
        json.dump(node_count_dic,f)


def three_in_one(csv_file_directory, valid_cate, relate_holder_diretory, see_also_holder_directory):
    """

    because the relate holder and see alos holder are in the same sequence of keys , so filter it in the same tiem.


    :param csv_file_direcotry: the csv file that store the valdate keyword after valid cate filtering.
    :param valid_cates: the category that wanted, a list.
    :param relate_holder_diretory_holder: the related holder file path, in json format
    :param see_also_directory_holder: the see also holder. in json format
    :return:
    """
    files = list_full_paths('../wiki')
    new_relate_holder = dict()
    new_see_also_holder = dict()
    with open(csv_file_directory, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', )
        csv_writer.writerow(['#item#'])
        for file in tqdm(files):
            print(file)
            if not file.endswith('.ipynb_checkpoints') and not file.endswith('.ipynb'):
                with open(file, 'r') as f:
                    doc = json.load(f)
                    cates = doc['cate_holder']
                    relates = doc['relate_holder']
                    see_alsos = doc['see_also_holder']
                    i = 0
                    for key, values, relate, see_also in zip(cates.keys(),cates.values(), relates.values(), see_alsos.values()):
                        if type(key) is not str:
                            continue
                        else:
                            boole = re.match('.+:.+', key)
                            if boole is not None:
                                continue
                        if values is not None:
                            for value in values:
                                if value in valid_cate:



                                    csv_writer.writerow([key])
                                    # set to filter unneccessary
                                    new_relate_holder[key] = list(set(relate))
                                    new_see_also_holder[key] = list(set(see_also))

                                    break
    with open(relate_holder_diretory, 'w') as f:
        json.dump(new_relate_holder, f)
    with open(see_also_holder_directory, 'w') as f:
        json.dump(new_see_also_holder, f)



def get_node_degree_set_split(word2id_path,relate_holder_id_added,see_also_holder_id_added,node_degree_set_json,start,end):
    """
    too big,  split the word id into parts to write in seperate files.

    :param word2id_path:
    :param relate_holder_id_added:
    :param see_also_holder_id_added:
    :param node_degree_set_json:
    :param start:
    :param end:
    :return:
    """
    with open (word2id_path,'r',encoding='utf-8') as f:
        dic=json.load(f)
    length= len(dic)
    node_count_dic=dict()
    for i in range(length):
        node_count_dic[i]=[]
    with open(relate_holder_id_added, 'r', encoding='utf-8') as f:
        relates = json.load(f)
        count = 0
        print('total:{}'.format(len(relates)))
        for value in tqdm(relates.values()):

            x = len(value)

            if x > 0:
                for i in range(0, x):
                    if start<value[i]<=end:
                        node_count_dic[value[i]]=list(set(node_count_dic[value[i]]+list(value)))

            count = count + 1
            if count > 10000 and count % 10000 == 0:
                print(count)

    with open(see_also_holder_id_added, 'r', encoding='utf-8') as f:
        see_alsos= json.load(f)
        count = 0
        for value in see_alsos.values():

            x = len(value)

            if x > 0:

                for i in range(0, x):
                    if start<value[i]<=end:
                        node_count_dic[value[i]]=list(set(node_count_dic[value[i]]+list(value)))

            count = count + 1
            if count > 10000 and count % 10000 == 0:
                print(count)
    with open(node_degree_set_json,'w',encoding='utf-8') as f:
        json.dump(node_count_dic,f)

def transfer_node_degree_to_with_words_sorted(node_degree_path,id2word_path,node_degree_words_sorted_path):

    """

    have a look at what the id2node_degree --> word2node_degree

    :param node_degree_path:
    :param id2word_path:
    :param node_degree_words_sorted_path:
    :return:
    """

    with open (node_degree_path,'r',encoding ='utf-8') as f:
        node_degree=json.load(f)
    node_degree_list  = list(node_degree.values())
    with open (id2word_path,'r',encoding  ='utf-8') as f:
        id2word =json.load(f)
    words  = list(id2word.values())
    del node_degree_list[0]
    del words[-1]
    node_degree_with_words  =dict(zip(words,node_degree_list))
    node_degree_with_words_sorted=sorted(node_degree_with_words.items(), key=lambda item: item[1], reverse=True)

    with open (node_degree_words_sorted_path,'w',encoding ='utf-8') as f:
        json.dump(node_degree_with_words_sorted,f)



def histogram_plot(node_degree_path,x_end,bins_number,y_end):
    """

    plot the histogram of the node_degree,
    x axis is the node degree
    y axis is the count

    :param node_degree_path:  the file path of the node degree
    :param x_end:  the x axis max value
    :param bins_number:  the number of bins of node_degree, in x axis.
    :param y_end:  the max value of count, ye degree
    :return:  no return , the plot will be shown
    """

    with open(node_degree_path, 'r', encoding='utf-8') as f:
        node_degree = json.load(f)
    node_degree_list = list(node_degree.values())

    del node_degree_list[0]
    #  0 is UNK, and abnormal super big, which is unneccessay for the process.
    node_degree_array = np.array(node_degree_list)

    bins = np.linspace(0, x_end, bins_number)
    plt.figure(figsize=(24, 8))

    plt.xlim([min(node_degree_array) - 5, x_end])
    plt.ylim(0, y_end)
    plt.hist(node_degree_array, bins=bins, alpha=0.5)

    plt.title('Node_degree')
    plt.xlabel('variable X')
    plt.ylabel('count')

    plt.show()



def w2w_further_processing_split():
    """
    after filtering with node degree, we get a w2w file,
    however the file is still too big for edge weight filtering,  so it is splitted into parts
    and doing the sum up in parts
    :return:
    """
    from collections import Counter
    rows= []
    for i in range (0,27):
        rows.append((i)*150000000)
    for row in rows:
        c = Counter()
        if row == rows[-1]:
            break
        count = 0
        path_name = 'all_cate_l_3/relation_split/w2w_'+str(row)+'.csv'
        for df in pd.read_csv('all_cate_l_3/w2w_need_further_processing_220.csv', sep=';', header=None,skiprows=row,
                              chunksize=10000000, names=['head', 'tail', 'count']):
            count =count+1
            print ("processing {} part about {} %:".format(row/150000000,count /15*100))
            c.update(df.groupby(['head', 'tail'])['count'].sum().to_dict())

            if count==15:
                break



        df = pd.DataFrame.from_dict(c, orient='index', columns=['count'])
        mi = pd.MultiIndex.from_tuples(df.index, names=['head', 'tail'])
        df = df.set_index(mi).reset_index()
        df.to_csv(path_name,sep=';',header = False,index = False)




def delete_all_0_in_split_files():
    """
    before continue to combination, do the 0 ['UNK'] node deleteion first
    :return:
    """
    split_files = list_full_paths('all_cate_l_3/relation_split')
    for file in tqdm(split_files):

        df1 = pd.read_csv(file, sep=';', header=None, names=['head', 'tail', 'count'])
        print (len(df1))
        df1= df1[df1['head']!=0]
        print (len (df1))
        df1.to_csv(file,sep=';',header = False,index = False)



def to_filter_to_have_XX_left_and_also_filter_the_key_and_values():
    """
    filter based on the node degree
    :return:
    """
    with open('all_cate_l_3/node_degree_split/node_degree_all_sorted.json' , 'r',encoding ='utf-8') as f:
        dict = json.load (f)
    to_be_del = []
    for key,value in dict.items():
        if 0<= value <220:
            to_be_del.append(int(key))

    with open ('all_cate_l_3/relate_holder_id_added.json','r',encoding ='utf-8') as f, open('all_cate_l_3/see_also_holder_id_added.json','r',encoding ='utf-8') as m:
        relates = json.load(f)
        see_alsos =  json.load(m)
    print (len (relates))
    print (len (see_alsos))

    for key in tqdm(to_be_del):
        if str(key) in relates.keys():
            del relates [ str(key)]


            del see_alsos[str(key)]
    print (len (relates))
    print (len (see_alsos))
    for values in tqdm(relates.values()):
        if len(values)<220:
            values.sort()
            values.reverse()
            idx = 0
            for value in values[::-1]:
                while value > to_be_del[idx]:
                    if idx == len(to_be_del)-1:
                        break
                    idx = idx+1

                if value == to_be_del[idx]:
                    values.remove(value)
                    if idx ==len(to_be_del)-1:
                        break
                    idx=idx+1

    for values in tqdm(see_alsos.values()):
        if len (values)<220:
            values.sort()
            values.reverse()
            idx = 0
            for value in values[::-1]:
                while value > to_be_del[idx]:
                    if idx == len(to_be_del)-1:
                        break
                    idx = idx+1

                if value == to_be_del[idx]:
                    values.remove(value)
                    if idx ==len(to_be_del)-1:
                        break
                    idx=idx+1
    with open('all_cate_l_3/relate_holder_id_added_filtered_with_node_degree_threshold_1000.json','w', encoding='utf-8') as f:
        json.dump(relates,f)
    with open('all_cate_l_3/see_also_holder_id_added_filtered_with_node_degree_threshold_1000.json','w', encoding='utf-8') as f:
        json.dump(see_alsos,f)



def processing_combination():
    """
    after splitting to parts and do the sum up in each parts,
    recombine it into bigger parts and do sum up again


    1000 lines --> 10 parts *100 --> 10parts *50 --> 1 part *500 -->1 part *240

    :return:
    """
    split_files = list_full_paths('all_cate_l_3/relation_split')
    i = 0
    while i<=len(split_files)-1:
        print ('procesisng {} %'.format(i/26*100))
        if i== 24:
            df1 = pd.read_csv(split_files[i], sep=';', header=None, names=['head', 'tail', 'count'])
            df2 = pd.read_csv(split_files[i + 1], sep=';', header=None, names=['head', 'tail', 'count'])

            i = i + 2
            print("total len: {}".format(len(df1) + len(df2) ))
            result = df1.append(df2)
            print('clear start')
            del df1, df2
            print('clear end')
            result = result.groupby(['head', 'tail'])['count'].sum().reset_index()
            print(len(result))
            name = 'all_cate_l_3/relation_comb/w2w_combination_' + str(i) + '.csv'
            result.to_csv(name, sep=';', header=False, index=False)
            break

        df1 = pd.read_csv(split_files[i], sep=';', header=None, names=['head', 'tail', 'count'])
        df2 = pd.read_csv(split_files[i+1], sep=';', header=None, names=['head', 'tail', 'count'])
        df3 = pd.read_csv(split_files[i+2], sep=';', header=None, names=['head', 'tail', 'count'])
        i=i+3

        result = df1.append([df2,df3])

        del df1,df2,df3

        result = result.groupby(['head', 'tail'])['count'].sum().reset_index()

        name = 'all_cate_l_3/relation_comb/w2w_combination_'+ str(i)+'.csv'
        result.to_csv(name,sep=';',header = False,index = False)
        del result
        print ('end')


def node_degree_plot(list,x_end,bins_number,y_end):


    node_degree_list = list

    # del node_degree_list[0]
    #  0 is UNK, and abnormal super big, which is unneccessay for the process.
    node_degree_array = np.array(node_degree_list)

    bins = np.linspace(0, x_end, bins_number)
    plt.figure(figsize=(24, 8))

    plt.xlim([min(node_degree_array) - 5, x_end])
    plt.ylim(0, y_end)
    plt.hist(node_degree_array, bins=bins, alpha=0.5)

    plt.title('node_degree')
    plt.xlabel('variable X')
    plt.ylabel('count')

    plt.show()

if __name__ == '__main__':

    # floor=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    # floor =[i*100000 for i in floor]
    #
    # for i in range((len(floor)-1)):
    #     node_degree_set_json  = 'all_cate_l_3/node_degree_split/node_degree_set_split_'+str(i+1)+'.json'
    #     get_node_degree_set_split('all_cate_l_3/word2id.json', 'all_cate_l_3/relate_holder_id_added.json',
    #                         'all_cate_l_3/see_also_holder_id_added.json',
    #                          node_degree_set_json,floor[i],floor[i+1])

    #  !!!!!!!!!!!!!!!!!!!!!attention that word2id and id2word put the ['UNK'] : 0 at last !!!!!!!!!!!!!!!!

    # following is to process all


    # all_categories_3 = list_full_paths("../all_categories_3")
    # print(all_categories_3)
    #
    # all_categories_3_short = os.listdir("../all_categories_3")
    #
    # all_cate = []
    # for category in zip(all_categories_3, all_categories_3_short):
    #     if category[1].endswith('.ipynb_checkpoints') is False:
    #
    #         valid_cate = read_technology_categories(category[0])
    #         cate_name, _ = os.path.splitext(category[1])
    #         dir_path = '../' + cate_name
    #         os.makedirs(dir_path)
    #         read_technology_related_items(dir_path + '/' + cate_name + '_keyword_l3.csv', valid_cate)
    #         per = filtered_meaning_less(dir_path + '/' + cate_name + '_keyword_l3.csv',
    #                               dir_path + '/' + cate_name + '_filtered_keyword_l3.csv')
    #
    #         all_cate = all_cate + per
    # all_cate = set(all_cate)
    #
    # with open("all_cates_3_filtered_list.csv", 'r', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     for keyword in all_cate:
    #         writer.writerow([keyword])

            # relate_holder_filter (dir_path+'/'+cate_name+'_filtered_keyword_l3.csv',dir_path+'/'+cate_name+'_filtered_relate_holder_l3.json')
            #
            # see_also_holder_filter (dir_path+'/'+cate_name+'_filtered_keyword_l3.csv',dir_path+'/'+cate_name+'_filtered_see_also_holder_l3.json')
    #
    # df = pd.read_csv("all_cate_l_3/all_cates_3_filtered_list.csv",names=['#item#'])






    # ------ after having the links, to prepare until w2w-----

    #
    # relate_holder_filter_not_sequential('all_cate_l_3/all_cates_3_filtered_list.csv','all_cate_l_3/all_relate_holder.json')
    # #
    # see_also_holder_filter_not_sequential('all_cate_l_3/all_cates_3_filtered_list.csv','all_cate_l_3/all_see_also_holder.json')

    # 3in1 to process all


    # all_categories_3 = list_full_paths("../all_categories_3")
    #
    #
    # all_categories_3_short = os.listdir("../all_categories_3")
    #
    # all_cate = []
    # for category in zip(all_categories_3, all_categories_3_short):
    #     if category[1].endswith('.ipynb_checkpoints') is False:
    #         print (category[0])
    #         valid_cate = read_technology_categories(category[0])
    #         all_cate = all_cate+valid_cate
    # all_cate = set(all_cate)
    #
    # three_in_one('all_cate_l_3/filtered_keyword_l3.csv',all_cate,'all_cate_l_3/relate_holder.json','all_cate_l_3/see_also_holder.json')
    # #
    #
    # #
    #
    #
    # #
    # get_keywords_list('all_cate_l_3/filtered_keyword_l3.csv','all_cate_l_3/relate_holder.json','all_cate_l_3/see_also_holder.json','all_cate_l_3/keyword_list_l3.csv')
    # set_keywords_list('all_cate_l_3/keyword_list_l3.csv','all_cate_l_3/keyword_list_set_l3.csv')
    # filter_set_keywords_list('all_cate_l_3/keyword_list_set_l3.csv','all_cate_l_3/keyword_list_set_filtered_l3.csv')
    #
    # word2id_and_id2word_list('all_cate_l_3/keyword_list_set_filtered_l3.csv','all_cate_l_3/word2id.json','all_cate_l_3/id2word.json')
    #
    # with open('all_cate_l_3/relate_holder.json', 'r', encoding='utf-8') as f:
    #     relates = json.load(f)
    #     print(len(relates))
    #     del relates['NaN']
    #     print(len(relates))
    # with open('all_cate_l_3/relate_holder.json', 'w', encoding='utf-8') as f:
    #     json.dump(relates, f)
    # with open('all_cate_l_3/see_also_holder.json', 'r', encoding='utf-8') as f:
    #     see_alsos = json.load(f)
    #     print(len(see_alsos))
    #     del see_alsos['NaN']
    #     print(len(see_alsos))
    # with open('all_cate_l_3/see_also_holder.json', 'w', encoding='utf-8') as f:
    #     json.dump(see_alsos, f)
    #
    # word2id('all_cate_l_3/word2id.json','all_cate_l_3/relate_holder.json','all_cate_l_3/see_also_holder.json','all_cate_l_3/relate_holder_id.json','all_cate_l_3/see_also_holder_id.json')
    # adding_item_to_relate_and_see_also('all_cate_l_3/relate_holder_id.json','all_cate_l_3/see_also_holder_id.json','all_cate_l_3/relate_holder_id_added.json','all_cate_l_3/see_also_holder_id_added.json')
    # w2w_csv_too_big('all_cate_l_3/relate_holder_id_added_filtered_with_node_degree_threshold_220.json','all_cate_l_3/see_also_holder_id_added_filtered_with_node_degree_threshold_220.json','all_cate_l_3/w2w_need_further_processing_220.csv')
    # w2w_csv_every('all_cate_l_3/relate_holder_id_added_filtered_with_node_degree_threshold_1000.json','all_cate_l_3/see_also_holder_id_added_filtered_with_node_degree_threshold_1000.json','all_cate_l_3/w2w_need_further_processing.csv')

    # with open('all_cate_l_3/see_also_holder_id_added_filtered_with_node_degree_threshold_1000.json' ,'r', encoding='utf-8') as f:
    #     relates = json.load(f)
    #     count = 0
    #     print('total:{}'.format(len(relates)))

    # df =pd.read_csv ('all_cate_l_3/filtered_keyword_l3.csv')
    # df1 = list(df['#item#'])
    # df1.remove('NaN')
    # print (len(df))
    # print (len(df1))
    # with open(set_keywords_list_directory, 'w', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     for keyword in keywords_list_set:
    #         writer.writerow([keyword])






    # import numpy as np
    # import math
    # with open ('all_cate_l_3/node_degree_split/node_degree_all_sorted.json','r',encoding ='utf-8') as f:
    #     node_degree=json.load(f)
    # node_degree_list  = list(node_degree.values())
    # with open ('all_cate_l_3/id2word.json','r',encoding  ='utf-8') as f:
    #     id2word =json.load(f)
    # words  = list(id2word.values())
    #
    #
    # del node_degree_list[0]
    # del words[-1]
    # node_degree_with_words  =dict(zip(words,node_degree_list))
    #
    # node_degree_with_words_sorted=sorted(node_degree_with_words.items(), key=lambda item: item[1], reverse=True)
    # with open ('all_cate_l_3/node_degree_split/node_dgree_with_words_sorted.json','w',encoding ='utf-8') as f:
    #     json.dump(node_degree_with_words_sorted,f)
    #

    # import numpy as np
    # import math
    #
    # with open('all_cate_l_3/node_degree_split/node_degree_all_sorted.json', 'r', encoding='utf-8') as f:
    #     node_degree = json.load(f)
    # with open('all_cate_l_3/id2word.json', 'r', encoding='utf-8') as f:
    #     id2word = json.load(f)
    # words = list(id2word.values())
    # node_degree_with_words = dict()
    # for key, value in node_degree.items():
    #     if key!= '0':
    #         node_degree_with_words[str(id2word[key])]= value
    #
    #
    # node_degree_with_words_sorted = sorted(node_degree_with_words.items(), key=lambda item: item[1], reverse=True)
    # with open('all_cate_l_3/node_degree_split/node_dgree_with_words_sorted.json', 'w', encoding='utf-8') as f:
    #     json.dump(node_degree_with_words_sorted, f)






    # node_degree_list=sorted(node_degree_list)
#     # print (node_degree_list)
#     node_degree_array = np.array(node_degree_list)
#     import matplotlib.pyplot as plt
#     #
#     #
#
#     #
#     # bins = np.linspace(math.ceil(min(node_degree_array)),
#     #                    math.floor(max(node_degree_array)),
#     #                    20) # fixed number of bins
#     # bins  = compute_histogram_bins(node_degree_array,50)
#     bins = np.linspace(0, 1000, 100)
#
# # bins = [0,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,
#     #         50000,60000,70000,80000,90000,100000,1000000,6000000]
#
#     plt.xlim([min(node_degree_array)-5, 1000])
#     plt.ylim(0,100000)
#     plt.hist(node_degree_array, bins=bins, alpha=0.5)
#
#     plt.title('Random Gaussian data (fixed number of bins)')
#     plt.xlabel('variable X (20 evenly spaced bins)')
#     plt.ylabel('count')
#
#     plt.show()
#
#


    # test 5728533: uaw-gm teamwork 500"
    # with open ('all_cate_l_3/id2word.json','r',encoding='utf-8') as f:
    #     id2word= json.load(f)
    # with open('all_cate_l_3/relate_holder_id_added.json','r',encoding ='utf-8') as f:
    #     doc= json.load(f)
    # count = 0
    # for key,value in doc.items():
    #     x=len(value)
    #     for i in range (x):
    #         if value[i]==22143:
    #             print(x)
    #             print (id2word[key])
    #             count = count+x
    #             break
    # print (count)

    # histogram_plot('all_cate_l_3/node_degree_split/node_degree_all_sorted.json',500,500,55000)
    #
    # with open('all_cate_l_3/node_degree_split/node_degree_all_sorted.json' , 'r',encoding ='utf-8') as f:
    #     dict = json.load (f)
    # count = 0
    # for value in tqdm(dict.values()):
    #     if 220<= value <1540000:
    #         count =count+1
    # print (count)
    # print (count/7230294*100)

    # filter to have 43% left


    # df = pd.read_csv('all_cate_l_3/w2w_need_further_processing.csv')
    # print (len(df))
    # totally there are 32*10 ^9 lines in csv file


    # def big_csv_counting_after_w2w_too_big():
    #     from collections import Counter
    #     count =0
    #     c =Counter()
    #     for df in pd.read_csv('all_cate_l_3/w2w_need_further_processing.csv', sep=';', header=None, chunksize=10000000,names = ['head','tail','count']):
    #         c.update(df.groupby(['head', 'tail'])['count'].sum().to_dict())
    #         count =count+1
    #         print ("processing about {} %:".format(count /320*100))
    #
    #     df = pd.DataFrame.from_dict(c, orient='index', columns=['count'])
    #     mi = pd.MultiIndex.from_tuples(df.index, names=['head', 'tail'])
    #     df = df.set_index(mi).reset_index()
    #     print (df)
    #     df.to_csv('check_here.csv',sep=';',index =False)
    # print (count)
    #

    # processing_combination()
    #
    # y = len(result)
    #
    # print (y)
    # def do_combination_w2w():
    #     split_files = list_full_paths('all_cate_l_3/relation_split')
    #     i=0
    #     while count<2 :
    #         count = 0
    #         def create_df (number,i):
    #             name= 'df_'+str(i)
    #
    #         df1 = pd.read_csv(split_files[i], sep=';', header=None, names=['head', 'tail', 'count'])
    #         i=i+1
    #         df1 = pd.read_csv(split_files[i], sep=';', header=None, names=['head', 'tail', 'count'])
    #         i=i+1


    #
    # files = list_full_paths('all_cate_l_3/relation_split')
    # total_length=0
    #
    # for file in tqdm(files):
    #     df= pd.read_csv(file, sep=';', header=None, names=['head', 'tail', 'count'])
    #
    #     total_length= total_length+len(df)
    #     print (total_length)



    def filter_back (n):
        files = list_full_paths('all_cate_l_3/relation_split')

        df_new = pd.DataFrame()
        for file in tqdm(files):
            df = pd.read_csv(file, sep=';', header=None, names=['head', 'tail', 'count'])
            df_new = df_new.append(df[df['count']>=2])

        df_new.to_csv('all_cate_l_3/w2w_after_filter/w2w_2.csv',sep=';',index=False,header=False)
    # filter_back(2)
    def filter_back_then_filter (file):

        df = pd.read_csv(file, sep=';', header=None, names=['head', 'tail', 'count'])
        print (len(df))
        df = df.groupby(['head', 'tail'])['count'].sum().reset_index()
        print (len(df))
        df.to_csv('all_cate_l_3/w2w_after_filter/w2w_2_quchong.csv',sep=';',index=False,header=False)
    # filter_back_then_filter('all_cate_l_3/w2w_after_filter/w2w_2.csv')
    # filter_back_then_filter('all_cate_l_3/w2w_after_filter/w2w_3.csv')
    # filter_back(4)
    # df = pd.read_csv('all_cate_l_3/w2w_after_filter/w2w_4.csv', sep=';', header=None, names=['head', 'tail', 'count'])
    # print (len(df))
    # df = pd.read_csv('all_cate_l_3/w2w_after_filter/w2w_3.csv', sep=';', header=None, names=['head', 'tail', 'count'])
    # print ( len(df))
    # edge= [2,3,4,5,6,7,8,9]
    # df = pd.read_csv('all_cate_l_3/w2w_after_filter/w2w_3_quchong.csv', sep=';', header=None, names=['head', 'tail', 'count'])
    # df1=df[df['head']==6533884]
    # df2=df[df['tail']==6533884]
    # df1= df1.append(df2)
    # df1.to_csv('all_cate_l_3/w2w_after_filter/w2w_artificial.csv', sep=';', index=False, header=False)
    #
    # print (len (df1))
    # print (len(df[df['head']==6533884])+(len(df[df['tail']==6533884])))
    # df = df[df['head']!=df['tail']]
    # df.to_csv('all_cate_l_3/w2w_after_filter/w2w_3_quchong.csv', sep=';', index=False, header=False)

    # for i in tqdm(edge):]
    #     print ('{} : number :{}'.format(i,len(df[df['count']==i])))
    # for i in tqdm(edge):
    #     print ('{} : number :{}'.format(i,len(df[df['count']==i])))

    def histogram_plot(csv,x_end,bins_number,y_end):

        df = pd.read_csv(csv, sep=';', header=None,
                         names=['head', 'tail', 'count'])
        node_degree_list = df['count'].tolist()

        del node_degree_list[0]
        #  0 is UNK, and abnormal super big, which is unneccessay for the process.
        node_degree_array = np.array(node_degree_list)

        bins = np.linspace(0, x_end, bins_number)
        plt.figure(figsize=(24, 8))

        plt.xlim([min(node_degree_array) - 5, x_end])
        plt.ylim(0, y_end)
        plt.hist(node_degree_array, bins=bins, alpha=0.5)

        plt.title('edge weight')
        plt.xlabel('variable X')
        plt.ylabel('count')

        plt.show()


    # histogram_plot('all_cate_l_3/w2w_after_filter/w2w_artificial.csv',150,100,400)
    # df_a = pd.read_csv('all_cate_l_3/w2w_after_filter/w2w_artificial.csv', sep=';', header=None,names=['head', 'tail', 'count'])
    # with open('all_cate_l_3/id2word.json','r',encoding='utf-8') as f:
    #     id2word=json.load(f)
    # df_a['head_mean']=None
    # df_a['tail_mean']= None
    # for i in range (len(df_a)):
    #     df_a['head_mean'].iloc[i] =  id2word[str(df_a['head'].iloc[i])]
    #     df_a['tail_mean'] .iloc[i] =  id2word[str(df_a['tail'].iloc[i])]
    # df_a.to_csv('all_cate_l_3/w2w_after_filter/w2w_artificial_meaning.csv', sep=';', index=False, header=False)





    # G = nx.read_gpickle('all_cate_l_3/w2w_after_filter/reduced_wiki_undirected_220_3_all.gpickle')
    # node_degree_list =[]
    # df = pd.DataFrame()
    # for i in tqdm(G.nodes()):
    #     node_degree_list.append(G.degree(i))
    # node_degree_plot(node_degree_list,10000,2000,1000)


    # with open('all_cate_l_3/w2w_after_filter/node_degree_list.csv', 'w', newline='') as f:
    #     csvwriter = csv.writer(f, delimiter=';')
    #
    #     for i in tqdm(G.nodes()):
    #         csvwriter.writerow([G.degree(i)])

    # df = pd.read_csv('all_cate_l_3/w2w_after_filter/node_degree_list.csv',names =['count'])
    # all_n = len(df)
    # df = df[df['count']==2]
    # df = df[df['count']==2]
    # print (len(df))
    # print (len(df)/all_n*100)

 # test G's 'artificial intellingece' node's disparity alpha process ,figure out why the assumption is not rght.
 #      thi is to read the all_cate_with_disparity
 #    G = nx.read_gpickle('all_cate_l_3/w2w_after_filter/disparity_filtered_calculated_wiki_undirected_220_3_all.gpickle')
    def test_disparity_filter(G, alpha_thred):
        """
        Network reduction by disparity filter.
        :param G: networkx graph to be reduced
        :param alpha_thred: the edges with maxAlpha >= alpha_thred will be preserved
        :return: G
        """
        # disparity_filter
        all_edges = G.edges(6533884)
        for (a, b) in all_edges:
            if G[a][b]['maxAlpha'] < alpha_thred:
                G.remove_edge(a, b)
        print('edges: ', len(G.edges()))
        print('nodes: ', len(G.nodes()), '\n')

        return G


    def test_disparity_alpha(G):
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
        all_edges_with_weight = G.edges(6533884,data='weight')
        # for (a, b, w) in tqdm(G.edges(data='weight')):
        for (a, b, w) in tqdm(all_edges_with_weight):

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

        print('edges: ', len(G.edges()))
        print('nodes: ', len(G.nodes()), '\n')

        return G

    # test_disparity_filter(G,0.99)
    #  test_disp_calculation
    G = nx.read_gpickle('all_cate_l_3/w2w_after_filter/reduced_wiki_undirected_220_3_all.gpickle')
    test_disparity_alpha(G)
