import threading
import time
import PF
import json
from random import randint


def error_parameters(G, query_type, parameters):
    if query_type == 'get_Rel_one':
        start_id = parameters['ipt']
        start_label = G.node[start_id]["label"]
        end_id = 'null'
        end_label = 'null'
    elif query_type == 'find_paths':
        start_id = min(parameters['source'], parameters['target'])
        end_id = max(parameters['source'], parameters['target'])
        start_label = G.node[start_id]["label"]
        end_label = G.node[end_id]["label"]
    elif query_type == 'find_paths_clusters':
        start_id = parameters['cluster1']
        end_id = parameters['cluster2']
        start_label = [G.node[n]['label'] for n in start_id]
        end_label = [G.node[n]['label'] for n in end_id]
    else:
        raise TypeError('unknown query_type')

    return start_id, start_label, end_id, end_label


def create_userInteractData_Table(schema):
    """
    create a table to store the activities of the user
    1. only store the user data when using Global setting.
    2. If type is explore, save the original record order.
    If primary key is same, assert position is same, count +1
    3. If type is find path, save the path from small id to bigger id.
    If primary key is same, assert position is same, count +1
    4. If type is B path, save the path from small id to bigger id.
    If primary key is same, update the position to later position, count +1
    5. If type is search, record_wid is the id of the word, record_label is the label of the search word.
    position always remain to 1, count +1

    create another table to store the error or empty result of the user
    1. If type is search button, start_label is the user's query. count+1
    2. If type is explore, start_label is the user's query and start_id is the wid of this query. count+1
    3. If type is path, start_id is the smaller id, start_label is the word label of smaller id. End_id is the larger id,
        End_label is the word label of larger. count +1
    4. If type is Bpath, start_id is the id list of cluster1, start_label are the list of corresponding labels.
        end_id is the id list of cluster2, end_label are the list of corresponding labels. count +1

    :param schema: is the schema to create the table
    :return:
    """
    cnx, cursor = PF.creatCursor(schema, 'W')
    Qy = ("""
        create table `user_record`
        (
            `data_version` varchar(200) not null,
            `distance_type` varchar(200) not null,
            `eid` varchar(200) not null,
            `query_type` varchar(200) not null,
            `record_wid` varchar(18000) not null,
            `record_label` varchar(3000) not null,
            `position` int unsigned not null,
            `count` int default 0 not null,
            `datetime` DATETIME not null,
            primary key (`data_version`(100),`distance_type`(100),`eid`(100),`query_type`(100),`record_wid`(255)),
            index(`distance_type`),
            index(`eid`),
            index(`query_type`),
            index(`position`),
            index(`count`),
            index(`datetime`)
        )
        """)

    cursor.execute(Qy)

    errorQy = ("""
            create table `user_error`
            (
                `data_version` varchar(200) not null,
                `distance_type` varchar(200) not null,
                `eid` varchar(200) not null,
                `query_type` varchar(200) not null,
                `start_id` varchar(300) not null,
                `start_label` varchar(1000) not null,
                `end_id` varchar(300) not null,
                `end_label` varchar(1000) not null,
                `count` int default 0 not null,
                `datetime` DATETIME not null,
                primary key (`data_version`(64),`distance_type`(64),`eid`(64),`query_type`(64),`start_id`(255),`start_label`(255),`end_id`(255)),
                index(`distance_type`),
                index(`eid`),
                index(`query_type`),
                index(`count`),
                index(`datetime`)
            )
            """)
    cursor.execute(errorQy)

    cnx.commit()
    cursor.close()
    cnx.close()

    return


class record_thread(threading.Thread):
    def __init__(self, userSchema, data_version, distance_type, user, query_type, record_wids, record_labels, position):
        threading.Thread.__init__(self)
        self.userSchema = userSchema
        self.cnx, self.cursor = PF.creatCursor(self.userSchema, 'W')
        self.data_version = data_version
        self.distance_type = distance_type
        self.eid = user
        self.query_type = query_type
        self.record_wids = record_wids[0:]
        self.record_labels = record_labels[0:]
        self.position = position

    def run(self):
        for i, record in enumerate(self.record_wids):
            if self.query_type == 'search' or self.query_type == 'get_Rel_one' or self.query_type == 'generateClusters':
                record_wid = record
                record_label = self.record_labels[i]
                current_position = self.get_position(record_wid)
                print("current_position: ", current_position)
                newposition = self.position + i
                print("newposition: ", newposition)
                if current_position == -1:
                    self.insert_newline(record_wid, record_label, newposition)
                else:
                    assert current_position == newposition, 'position does not remain same in the explore or search function'  # Minimum Steps (minhops) may cause this error
                    self.addCount_updateTime(record_wid)

            elif self.query_type == 'find_paths':
                record_wid = record
                record_label = self.record_labels[i]
                if record_wid[0] > record_wid[-1]:
                    record_wid = record_wid[::-1]
                    record_label = record_label[::-1]
                current_position = self.get_position(record_wid)
                newposition = self.position + i
                if current_position == -1:
                    self.insert_newline(record_wid, record_label, newposition)
                else:
                    assert current_position == newposition, 'position does not remain same in the find_paths function'
                    self.addCount_updateTime(record_wid)

            elif self.query_type == 'find_paths_clusters':
                record_wid = record
                record_label = self.record_labels[i]
                if record_wid[0] > record_wid[-1]:
                    record_wid = record_wid[::-1]
                    record_label = record_label[::-1]
                current_position = self.get_position(record_wid)
                newposition = self.position + i
                if current_position == -1:
                    self.insert_newline(record_wid, record_label, newposition)
                else:
                    if newposition > current_position:
                        self.update_position(record_wid, newposition)
                    self.addCount_updateTime(record_wid)

        self.cnx.commit()

    def get_position(self, record_wid):
        Qy = ("""
                SELECT `position` from `user_record` where `data_version`=\'{}\' and `distance_type`=\'{}\' and `eid`=\'{}\' and `query_type`=\'{}\' and `record_wid`=\'{}\'
        """.format(self.data_version, self.distance_type, self.eid, self.query_type, json.dumps(record_wid)))
        print(Qy)
        self.cursor.execute(Qy)
        position = self.cursor.fetchall()
        assert len(position) <= 1, 'more than one position found, duplicated primary key'
        if len(position) == 1:
            return position[0][0]
        else:
            return -1

    def insert_newline(self, record_wid, record_label, position):
        Qy = ("""
            insert into `user_record` (`data_version`, `distance_type`, `eid`,`query_type`,`record_wid`,`record_label`,`position`,`count`,`datetime`)
            VALUES (\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',{},1,\'{}\')
        """.format(self.data_version, self.distance_type, self.eid, self.query_type, json.dumps(record_wid),
                   json.dumps(record_label), position, time.strftime('%Y-%m-%d %H:%M:%S'))
              )
        self.cursor.execute(Qy)

    def addCount_updateTime(self, record_wid):
        Qy = ("""
            update `user_record`
            set
            `count`=`count`+1,
            `datetime` = \'{}\'
            where `data_version`=\'{}\' and `distance_type`=\'{}\' and `eid`=\'{}\' and `query_type`=\'{}\' and `record_wid`=\'{}\'
        """.format(time.strftime('%Y-%m-%d %H:%M:%S'), self.data_version, self.distance_type, self.eid, self.query_type,
                   json.dumps(record_wid)))

        self.cursor.execute(Qy)

    def update_position(self, record_wid, position):
        Qy = ("""
            update `user_record`
            set `position`={}
            where `data_version`=\'{}\' and `distance_type`=\'{}\' and `eid`=\'{}\' and `query_type`=\'{}\' and `record_wid`=\'{}\'
        """.format(position, self.data_version, self.distance_type, self.eid, self.query_type, json.dumps(record_wid)))

        self.cursor.execute(Qy)


class error_thread(threading.Thread):
    def __init__(self, userSchema, data_version, distance_type, eid, query_type, start_id, start_label, end_id,
                 end_label):
        threading.Thread.__init__(self)
        self.userSchema = userSchema
        self.cnx, self.cursor = PF.creatCursor(self.userSchema, 'W')
        self.data_version = data_version
        self.distance_type = distance_type
        self.eid = eid
        self.query_type = query_type
        self.start_id = start_id
        self.start_label = start_label
        self.end_id = end_id
        self.end_label = end_label

    def run(self):
        Qy = ("""
            INSERT INTO `user_error` (`data_version`,`distance_type`,`eid`,`query_type`,`start_id`,`start_label`,`end_id`,`end_label`,`count`,`datetime`)
            VALUES (\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\',1, \'{}\')
            ON DUPLICATE KEY UPDATE `count`=`count`+1, `datetime`=\'{}\'
        """.format(self.data_version, self.distance_type, self.eid, self.query_type, json.dumps(self.start_id),
                   json.dumps(self.start_label), json.dumps(self.end_id), json.dumps(self.end_label),
                   time.strftime('%Y-%m-%d %H:%M:%S'), time.strftime('%Y-%m-%d %H:%M:%S'))
              )

        self.cursor.execute(Qy)
        self.cnx.commit()


def userQuestion(schema, user, N):
    """
    generate questions from record schema for user to answer
    :param user: the email id of a user
    :param N: the number of questions to be generated for each function/feature
    :param schema: the schema which records users' usage data

    :return: dictionary of the generated questions
    """

    def divide_range(R, x):
        """
        Seperate [0,R-1] into x sections. then randomly select one elements from each section
        :param R: range [0,R-1]
        :param x: number of section
        :return: the list of selected number
        """
        assert R >= x, 'Not Enought Range to be divided'
        rand_index = []
        for i in range(0, x):
            ns = int(round(float(R) / x * i))
            ne = int(round(float(R) / x * (i + 1))) - 1
            n = randint(ns, ne)
            rand_index.append(n)
        return rand_index

    def divide_list(array, x):
        """
        Cut array into x equal section, random select an element from each section
        :param array: input array
        :param x: x sections
        :return: cleaned array
        """
        nay = len(array)
        ind = divide_range(nay, x)
        newarry = []
        for i in ind:
            newarry.append(array[i])
        return newarry

    def filterlist(array):
        """
        remove duplicated element from list, while preserve the order
        :param array: input array
        :return: filter array
        """
        filterar = []
        for el in array:
            if el not in filterar:
                filterar.append(el)
        return filterar

    cnx, cursor = PF.creatCursor(schema, 'R')
    questions = {}
    N1 = N * 2

    for query_type in ['get_Rel_one', 'find_paths', 'find_paths_clusters', 'generateClusters']:
        Qy = ("""
            select `record_label` from `user_record` where `eid`=\'{}\' and `query_type`=\'{}\' order by `position` asc, `count` desc;
        """.format(user, query_type)
              )
        cursor.execute(Qy)
        results = cursor.fetchall()
        n_res = len(results)
        if n_res == 0:
            continue
        elif n_res <= N1:
            selected_index = [i for i in range(0, n_res)]
        else:
            selected_index = divide_range(n_res, N1)

        questions[query_type] = []

        for ind in selected_index:
            record = json.loads(results[ind][0])
            if query_type == 'get_Rel_one' and len(record) > 2:
                records = zip(record[0:-1], record[1:])
                for rd in records[0:2]:
                    Rd = '[' + '] -- ['.join(rd) + ']'
                    questions[query_type].append(Rd)
            else:
                record = '[' + '] -- ['.join(record) + ']'
                questions[query_type].append(record)

    ## remove duplicate element and get final results
    for key in questions.keys():
        questions[key] = filterlist(questions[key])
        n_rem = len(questions[key])
        if n_rem <= N:
            continue
        else:
            questions[key] = divide_list(questions[key], N)

    return questions


class mythread1(threading.Thread):
    def __init__(self, a, count):
        threading.Thread.__init__(self)
        self.count = count
        self.a = a

    def run(self):
        time.sleep(1.1)
        for i in range(0, self.count):
            time.sleep(1)
            self.a += 1
            print('sub: ', self.a)


class mythread2(threading.Thread):
    def __init__(self, a, count):
        threading.Thread.__init__(self)
        self.count = count
        self.a = a

    def run(self):
        time.sleep(1.2)
        for i in range(0, self.count):
            time.sleep(1)
            self.a += 1
            print('sub: ', self.a)


if __name__ == '__main__':
    a = 0
    # a.append(0)
    thread1 = mythread1(a, 5)
    thread2 = mythread2(a, 6)
    thread1.start()
    thread2.start()
    time.sleep(1)
    while True:
        time.sleep(1)
        print('main: ', a)
        if not thread1.isAlive() and not thread2.isAlive():
            break
