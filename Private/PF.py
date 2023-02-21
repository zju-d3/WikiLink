# Public Function for analysis
import mysql.connector
import pymysql
from nltk.stem import WordNetLemmatizer


# output the cursor of a schema
def creatCursor(shema, type):
    if type == "W":
        cnx = mysql.connector.connect(user='root', passwd='root', database=shema)
    elif type == "R":
        cnx = pymysql.connect(user='root', passwd='root', database=shema, charset='utf8')
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
    wl = WordNetLemmatizer()
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

    id = []
    for ipt in ipts:
        Qy = 'select `id` from `all_keywords` where `word`="{}"'.format(ipt)
        cursor.execute(Qy)
        try:
            id.append(cursor.fetchone()[0])
        except:
            raise TypeError("Input: {}, NOT Found".format(ipt))

    assert len(id) == len(ipts), "number of ids and corresponding inputs are not same"

    return id
