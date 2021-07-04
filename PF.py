# Public Function for analysis
import mysql.connector
import pymysql
from nltk.stem import WordNetLemmatizer
import numpy as np


# output the cursor of a schema
def creatCursor(shema, type):
    if type == "W":
        cnx = mysql.connector.connect(user='root', password='root', database=shema)
    elif type == "R":
        cnx = pymysql.connect(user='root', password='root', database=shema, charset='utf8')
    else:
        raise TypeError("No Type for Create Cursor")

    cursor = cnx.cursor()

    return cnx, cursor


# clean one phrase:
def lemma_onephrase(phrase):
    assert type(phrase) == str, 'input is not string'
    assert phrase != '', 'input is empty'

    phrase = phrase.lower()

    words = phrase.split(' ')
    wl = WordNetLemmatizer()  # delete something useless-->clean
    words = [wl.lemmatize(n) for n in words]
    cphr = ' '.join(words)
    return cphr


# clean inputs
# inputs is a list of words
def clean_inputs(ipts):
    cleaned = [lemma_onephrase(w) for w in ipts]

    return cleaned


# find ids of inputs
# inputs is a list of words
# checked OK
def find_id(ipts, cursor):
    ipts = clean_inputs(ipts)
    # print("find_id: ipts: ", ipts)
    id = []
    for ipt in ipts:
        Qy = 'select `id` from `all_keywords` where `word`="{}"'.format(ipt)
        cursor.execute(Qy)
        try:
            cursorData = cursor.fetchall()
            assert len(cursorData) <= 1, 'more than one row found for one word'
            id.append(cursorData[0][0])
        except:
            # print("PF_find_id_error: ", ipt)
            raise TypeError("Input: {}, NOT Found".format(ipt))

    assert len(id) == len(ipts), "number of ids and corresponding inputs are not same"
    # print("PF: find_id: id:", id)
    return id


def find_id_csv(ipts, word_map):
    ipts = clean_inputs(ipts)
    # print("find_id: ipts: ", ipts)
    id = []
    for ipt in ipts:
        id.append(int(word_map[ipt]))
    print(id)
    assert len(id) == len(ipts), "number of ids and corresponding inputs are not same"
    # print("PF: find_id: id:", id)
    return id, ipts


def genChain(*generators):
    """
    combine multiple generator into one
    the results of this output generator is the union of the results from all input generator
    :param generators: input generator
    :return: combined one generator
    """

    def funChain(x):
        elements = set()
        for it in generators:
            elements.update(list(it(x)))
        return iter(elements)

    return funChain


def listtopercentiles(ay):
    """
    map a list to its corresponding percentile by dictonary
    :param ay: original array
    :return: dictionary. Key is the element of array, value is the percentile
    """
    arry = sorted(ay)
    total = float(len(arry))
    percdict = {}
    for i, n in enumerate(arry):
        percdict.setdefault(n, []).append(i + 1)
    for key in percdict.keys():
        position = np.mean(percdict[key])
        percdict[key] = position / total
    return percdict


def scaling(maxV, minV, x):
    """
    scale v into [0,1] according to maxvalue and minvalue
    :param maxV: max value
    :param minV: min value
    :param x: x
    :return: scaled value
    """
    assert maxV > minV, 'max value should be larger than min value in feature scaling'
    assert x <= maxV, 'value not in range'
    assert x >= minV, 'value not in range'
    s = float((x - minV)) / (maxV - minV)
    return s
