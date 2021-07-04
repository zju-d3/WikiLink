import csv

from flask import Flask, render_template, make_response, request, session, redirect, url_for
import json
import networkx as nx
from networkanalysis.Analysis import Retrievor
from user_Feedback.recordUser import record_thread, error_thread, userQuestion
import PF
import time
from time import gmtime, strftime

app = Flask(__name__)
app.secret_key = '\x8b\x19\xa1\xb0D\x87?\xc1M\x04\xff\xc8\xbdE\xb1\xca\xe6\x9e\x8d\xb3+\xbe>\xd2'

# Initial Data
# whole retrievor, use whole database as its own graph
# myRtr = Retrievor.UndirectedG\
#     ('webapp/addNodeEdgeDegree_R+rn+GMHM_undirected_alpha0.65_nodeD1.0_total_v3_csvneo4j', 'total_v3_csvneo4j',
#      'userdata')

myRtr = Retrievor.UndirectedG('webapp/data2021/wiki_result_label_added_1_1', 'total_v3_csvneo4j', 'userdata')

# myRtr=Retrievor.UndirectedG('undirected(fortest)_R+G+SP+C+c','fortest','userdata')
# print('edges: ', len(myRtr.G.edges()))
# print('nodes: ', len(myRtr.G.nodes()))


with open('../webapp/data2021/wiki_word2id.json', encoding='utf8')as fp:
    word_map = json.load(fp)
    # print('这是文件中的json数据：', word_map)
    # print('这是读取到文件数据的数据类型：', type(word_map))


# print(word_map)

# sign up
@app.route('/signup')
def signup():
    print("signup")
    session['firstTimeVisit'] = True
    user = request.args.get('email', '')
    w = request.args.get('w', '')
    session['user'] = user
    session['w'] = w
    fusers = open('allusers.txt', mode='a')
    fusers.write(user + '\n')
    fusers.close()
    return redirect('/')


# Main Page
@app.route('/')
def index():
    print("index")
    if 'user' in session:
        print(session['w'])
        print(session['user'])
        w = session['w']
        if int(w) <= 750:
            print('moble')
            return make_response(open('m-index.html').read())
        else:
            return make_response(open('index.html').read())
    else:
        return make_response(open('signup.html').read())


@app.route('/mobile')
def mobile():
    return make_response(open('m-index.html').read())


# get text return nodes number
@app.route('/texttowid/<info>')
def texttowid(info):
    print("textword: info: ", info)
    info = json.loads(info)
    searchtext = info['searchtext']
    distance = 'None'
    # --new
    print("searchtext:", searchtext)
    ipts = [word.strip() for word in searchtext.split(';')]
    print("step1: texttowid: ipts:", ipts)  # ['apple', 'banana']
    try:
        # wids = myRtr.input_ids(ipts)
        wids, new_ipts = myRtr.input_ids_read_csv(word_map, ipts)
        print("texttowid: ", wids)
    except:
        # record the word which can't be found. error
        errthread = error_thread(myRtr.userSchema, myRtr.data_version, distance, session['user'], 'search', 'null',
                                 ipts[0], 'null', 'null')
        errthread.start()

        raise ValueError("Input NOT Found")

    labels = [myRtr.G.node[wid]['label'] for wid in wids]

    print("labels: ", labels)
    # labels = new_ipts

    # record user activity
    user = session['user']
    rthread = record_thread(myRtr.userSchema, myRtr.data_version, distance, user, 'search', wids, labels, 1)
    rthread.start()
    response = json.dumps(wids[-1])
    return make_response(response)


# search button, add one node
@app.route('/searchbutton/<info>')
def search(info):
    print("info:", info)
    info = json.loads(info)
    print("search: info: ", info)
    distance = info['tp']
    print("search: distance: ", distance)
    localG = myRtr.G.subgraph(set(info['currentnodes'] + info['query']))
    print(localG.nodes())
    allnodes = [
        # 每个顶点的次数（degree）
        {"wid": n, "label": localG.node[n]["label"], "N": localG.degree(n, weight="weight"), "n": localG.degree(n)} for
        n in localG.nodes()]
    alledges = [{"source": source, "target": target, 'dist': dist} for (source, target, dist) in
                localG.edges(data=distance)]
    sorted_paths = sorted(localG.edges(nbunch=info['query'], data=distance), key=lambda x: x[2])
    add_paths = [path[:-1] for path in sorted_paths]
    try:
        bi = 0
        bornnode = sorted_paths[bi][1]
        while bornnode in info['query']:
            bi += 1
            bornnode = sorted_paths[bi][1]
    except:
        bornnode = None

    dataset = {"allnodes": allnodes, "alledges": alledges, "paths": add_paths, 'bornnode': bornnode}
    print(dataset)
    response = json.dumps(dataset)
    return make_response(response)


# find the nearest node of the current nodes to the query node
@app.route('/findnear/<info>')
def findnear(info):
    info = json.loads(info)
    query = info['query']
    localG = myRtr.G.subgraph(set(info['currentnodes'] + [query]))
    sorted_neighbors = sorted([(n, localG[query][n]['Fw']) for n in localG.neighbors(query)], key=lambda x: x[1])
    try:
        bornnode = sorted_neighbors[0][0]
    except:
        bornnode = None

    response = json.dumps(bornnode)
    return make_response(response)


# Generate clusters based on current nodes
@app.route('/generateClusters/<info>')
def generateClusters(info):
    info = json.loads(info)
    nodes = info['nodes']
    method = info['method']
    weight = info['weight']

    # record clusters activities
    user = session['user']
    record_wid = sorted(nodes)
    rthread = record_thread(myRtr.userSchema, myRtr.data_version, weight, user, 'generateClusters', [record_wid],
                            ["Omit"], 1)
    rthread.start()

    if method == 'normalized':
        k = info['k']
        clusters = myRtr.cutgraph(nodes, k, weight=weight)

    elif method == 'mcl':
        r = info['r']
        M, clusters = myRtr.mcl_cluster(nodes, r, weight=weight)
    else:
        raise TypeError('unknown clustering method')

    # sort clusters by centrality
    distance = info['distance']
    clusters = myRtr.sort_clustersCentrality(clusters, distance)
    response = json.dumps(clusters)
    return make_response(response)


# query generator in the server
@app.route('/generator/<info>')
def generator(info):
    info = json.loads(info)
    distance = info['parameters']['parameters']['tp']
    query_type = info['parameters']['generator']

    info['parameters']['user'] = session['user']
    if info['explorelocal'] == True:
        info['localnodes'] = info['parameters']['parameters']['localnodes']

    explorenodes, explorepaths, position = myRtr.my_Gen(**info['parameters'])

    # record user activities
    if info['explorelocal'] == False:
        if len(explorepaths) > 0:
            record_wids = [path['ids'] for path in explorepaths]
            record_labels = [path['labels'] for path in explorepaths]
            rthread = record_thread(myRtr.userSchema, myRtr.data_version, distance, session['user'], query_type,
                                    record_wids, record_labels, position)
            rthread.start()

    if set(explorenodes).issubset(info['localnodes']):
        response = json.dumps({'AddNew': False, 'paths': explorepaths, "position": position})
    else:
        localG = myRtr.G.subgraph(set(info['localnodes']) | set(explorenodes))  # local
        allnodes = [
            {"wid": n, "label": localG.node[n]["label"], "N": localG.degree(n, weight="weight"), "n": localG.degree(n)}
            for n in localG.nodes()]
        alledges = [{"source": source, "target": target, 'dist': dist} for (source, target, dist) in
                    localG.edges(data=distance)]
        dataset = {'AddNew': True, "allnodes": allnodes, "alledges": alledges, "paths": explorepaths,
                   "position": position}
        response = json.dumps(dataset)

    return make_response(response)


# check first time visit
@app.route('/checkFirstTimevisit')
def checkFirstTimevisit():
    if session['firstTimeVisit'] == True:
        session['firstTimeVisit'] = False
        info = True
    else:
        info = False
    response = json.dumps(info)
    return make_response(response)


# get NeighborLevel for a node
@app.route('/neighbor_level/<int:node>')
def neighbor_level(node):
    response = my_localRtr.get_Neib_one(node, None)
    response['disconnected'] = set(my_localRtr.G.nodes()).difference(response["AllNb"])
    response.pop("AllNb")
    for key, value in response.iteritems():
        response[key] = list(value)

    response = json.dumps(response)
    return make_response(response)


# WordRank
# Get relevant word list and corresponding path list for a word
@app.route('/wordrank/<int:node>')
def wordrank(node):
    response = my_localRtr.get_Rel_one(node, "Fw", None)
    nodesandpaths = []
    for n, p in response.iteritems():
        path = p[1]
        nodesandpaths.append([n, path])
    response = json.dumps(nodesandpaths)
    return make_response(response)


############ FEEDBACK DATA COLLECTION #####################
@app.route('/feedback')
def feedback():
    return make_response(open('fb-completion.html').read())


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5000)

