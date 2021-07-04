# import urllib2
from lxml import html
import re
import mysql.connector
import nltk
# import enchant
from nltk.stem import WordNetLemmatizer
import string
from scrapy.loader.processors import Join
import pymysql
import datetime


# -----------------------------------------------------------------------------------
# input a list of words. Unicode, or string(ascii)
# output a list of lemmatized utf-8 words
# use default 'n' as postag
def lemma_listwords(keywords):  # OK No problem of this function
    # lowercase
    keywords = [n.lower() for n in keywords]

    s_join = Join()
    wl = WordNetLemmatizer()

    # seperate the word
    Seperate_words = []

    for n in keywords:
        # handle one keywords

        # define seperate punctuation

        # We still need to consider whether take ',' as the signal to seperate word
        n = n.replace('/', ',')
        n = n.replace('\\', ',')
        n = n.replace(';', ',')
        # split a word to a list of words sub_w
        sub_w = n.split(',')
        # filter empty element in the list
        sub_w = filter(None, [x.strip() for x in sub_w])

        # append each word in the list to Seperate_words
        for nonc in sub_w:
            # strip each word and append
            Seperate_words.append(nonc.strip())

    words = []

    for n in Seperate_words:
        # for one keywords

        # split one keywords to a list of word and punctuations: s
        s = nltk.word_tokenize(n)

        # lematize every word in the list
        for i, word in enumerate(s):
            s[i] = wl.lemmatize(word)
            if word == 'has':
                s[i] = 'have'

        joinw = s_join(s)  # lemmatized one single word

        for char in string.punctuation:
            joinw = joinw.replace(' ' + char, char)
            joinw = joinw.replace(char + ' ', char)

        words.append(joinw)

    return [wd.encode('utf-8') for wd in words]


# -----------------------------------------------------------------------------------------------------------------
# input a list of utf-8 items
# output a int list of corresponding id
def find_id(table, column, items, cursor):  # OK no problem!!!

    id = []
    for item in items:
        Qy = 'select `id` from `{}` where `{}`="{}"'.format(table, column, item)
        cursor.execute(Qy)
        id.append(cursor.fetchone()[0])

    assert len(id) == len(items), "number of items and corresponding items are not same"

    return id


# ---------------------------------------------------------------------
# input a schema
# output the cursor of this schema
def creatCursor(shema, type):  # OK no problem!!!

    if type == "W":
        cnx = mysql.connector.connect(user='wt', password='xx', database=shema)
    elif type == "R":
        cnx = pymysql.connect(user='rd', password='xx', database=shema, charset='utf8')
    else:
        raise TypeError("No Type for Create Cursor")

    cursor = cnx.cursor()

    return cnx, cursor


# ---------------------------------------------------------------
# handle w2w table
# input a list of int ids
# create their relationship based on string journal
def handle_w2w(Cursor, ids, table, journal):  # OK no problem!!!

    sids = sorted(ids)
    x = len(ids)

    for i in range(0, x - 1):
        for j in range(i + 1, x):
            RQy = ("""
            select EXISTS(SELECT rowid FROM {} WHERE rowid={} and colid={})
            """.format(table, sids[i], sids[j]))
            Cursor.execute(RQy)
            ex = Cursor.fetchone()[0]

            if ex == 0:
                WQy = ("""
                insert into {} values({},{},1,"{}")
                """.format(table, sids[i], sids[j], journal))

            else:
                WQy = ("""
                update
               `{}`
                set
               `value`=`value`+1,
               `journals`=if(`journals` REGEXP "(^{}$)|(^{}<>)|(<>{}$)|(<>{}<>)",
               `journals`,CONCAT(`journals`,"<>{}"))
                where rowid={} and colid={}
                """.format(table, journal, journal, journal, journal, journal, sids[i], sids[j]))

            Cursor.execute(WQy)
    return


# ------------------------------------------------------------------
# hand a2w table
# input list aids and list wids
def handle_a2w(Wcursor, aids, wids, tablename):  # ok no problem!!!
    for aid in aids:
        for wid in wids:
            WQy = ("""
             insert into `{}` (`aid`, `wid`, `value`) VALUES ({},{},1) ON DUPLICATE KEY UPDATE `value`=`value`+1
                 """.format(tablename, aid, wid))
            Wcursor.execute(WQy)

    return


# -----------------------------------------------------------------
# hand p2w table
# input list wids
# pid is a int
def handle_p2w(Wcursor, pid, wids, table):  # Ok no problem!!!
    for wid in wids:
        Qy = ("""
        INSERT ignore into `{}` (`pid`,`wid`) values ({},{})
            """.format(table, pid, wid))
        Wcursor.execute(Qy)

    return


# -----------------------------------------------------------------
# hand a2p table
# input list aids
# pid is a int
def handle_a2p(Wcursor, aids, pid, table):  # Ok no problem!!!
    for aid in aids:
        Qy = ("""
        INSERT ignore into `{}` (`aid`,`pid`) values ({},{})
            """.format(table, aid, pid))
        Wcursor.execute(Qy)

    return


# ----------------------------------------------------------------------
# save keywords
# keywords is a list of words in utf-8
def handle_allkeywords(Wcursor, keywords, tablename):  # OK no problem!!!

    for n in keywords:
        add_words = ('INSERT ignore into `{}` (`word`) values ("{}")'.format(tablename, n))
        Wcursor.execute(add_words)

    return


# ----------------------------------------------------------------------
# save authors
# authors is a list of authors in utf-8
def handle_allauthors(Wcursor, authors, tablename):  # OK no problem!!!

    for n in authors:
        add_authors = ('INSERT ignore into `{}` (`author`) values ("{}")'.format(tablename, n))
        Wcursor.execute(add_authors)

    return


# -----------------------------------------------------------------
# check existence of URL in MYsql whole table
def checkextc(tablename, Wcursor, URL):  # OK no problem!!!
    Qcheck = (""" select `id` from `{}` where `URL`="{}" """.format(tablename, URL))
    Wcursor.execute(Qcheck)
    row = Wcursor.fetchone()

    if row is None:
        return False
    else:
        return True


# ------------------------------------------------------------------
# handle whole table for a piece information
# all input information is UTF-8 except date is date data type
# output ids of words, authors and paper
def Allsql_onepaper(tablename, Wcursor, item):
    # insert whole table
    addrow = ("""
            INSERT into `{}` (`URL`,`title`,`Author`,`Journal`,`date`,`Keytext`) values
                                       ("{}","{}","{}","{}","{}","{}")
            """.format(tablename, item['URL'], item['title'], item['authortext'], item['journal'], item['date'],
                       item['keytext']))
    Wcursor.execute(addrow)

    # do all keywords
    handle_allkeywords(Wcursor, item['keywords'], 'all_keywords')
    # all authors
    handle_allauthors(Wcursor, item['authors'], 'all_authors')

    # find ids
    wids = find_id('all_keywords', 'word', item['keywords'], Wcursor)
    aids = find_id('all_authors', 'author', item['authors'], Wcursor)
    pid = find_id('whole', 'URL', [item['URL']], Wcursor)[0]

    # p2w
    handle_p2w(Wcursor, pid, wids, 'all_p2w')
    # a2p
    handle_a2p(Wcursor, aids, pid, 'all_a2p')
    # a2w
    handle_a2w(Wcursor, aids, wids, 'all_a2w')
    # w2w
    handle_w2w(Wcursor, wids, 'all_w2w', item['journal'])

    return aids, wids, pid
