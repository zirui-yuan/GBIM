import random
import numpy as np
from tqdm import tqdm
import time
import multiprocessing as mp
import math
# import GPy
import numpy as np
import heapq

def random_seeds(graph, k):
    user_num = graph.user_num
    item_num = graph.item_num
    candi_users = list(range(user_num))
    candi_items = list(range(item_num))
    seed_users = random.sample(candi_users, k)
    seed_items = random.sample(candi_items, k)
    seeds = set()
    for i in range(k):
        seeds.add((seed_users[i], seed_items[i]))
    return seeds

def get_k_max_positions(user_item_score, seeds, userids, itemids, k):
    max_positions = []
    for i in range(k):
        # 找到剩余部分中的最大值位置
        max_position_local = np.unravel_index(user_item_score[np.ix_(userids, itemids)].argmax(), (len(userids), len(itemids)))
        # 转化为全局位置

        max_position_global = (userids[max_position_local[0]], itemids[max_position_local[1]])
        # 存储最大值位置
        max_positions.append(max_position_global)
        # 剩余部分更新，去掉对应的行和列
        userids = np.delete(userids, max_position_local[0])
        itemids = np.delete(itemids, max_position_local[1])
    # seeds.update(max_positions)
    # seeds.append(max_positions)
    return max_positions

def maxdegree(graph, k):
    n = graph.user_num
    m = graph.item_num
    userids = list(range(n))
    itemids = list(range(m))
    user_item_score = np.zeros((n,m))
    seeds = []
    for i in range(n):
        user_degree = graph.get_out_degree('user', i)
        for j,item in enumerate(graph.item_list):
            item_degree = graph.get_out_degree('item', item)
            user_item_score[i][j] = user_degree*item_degree
    return get_k_max_positions(user_item_score, seeds, userids, itemids, k)
        
class generate_rr:
    def __init__(self,graph) -> None:
        self.graph = graph

    def generate_rr_lt(self, node_item_pair):
        # calculate reverse reachable set using LT model
        # seeds = set()
        # seeds.add(node_item_pair)
        # activity_nodes = self.model.calc_profit(seeds=seeds, return_nodes = True)
        # activity_nodes = list(activity_nodes)
        activity_nodes = list()
        activity_nodes.append(node_item_pair)

        
        activity_set = node_item_pair

        while activity_set != -1:
            user = activity_set[0]
            item_id = activity_set[1]
            activity_set = -1


            user_next = self.graph.get_neighbors(user)
            item_next = self.graph.get_item_neighbors(self.graph.item_list[item_id])
            if len(user_next) == 0 or len(item_next) == 0:
                break
            


            user_candidate = random.sample(user_next, 1)[0][0]
            item_candidate_id = self.graph.item_list.index(random.sample(item_next, 1)[0][0])

            if random.random()<0.5:
                if (user_candidate,item_id ) not in activity_nodes:
                    activity_nodes.append((user_candidate,item_id ))
                    activity_set = (user_candidate,item_id )
            else:
                if (user,item_candidate_id) not in activity_nodes:
                    activity_nodes.append((user,item_candidate_id))
                    activity_set = (user,item_candidate_id)
        return activity_nodes

    def generate_rr_ic(self, node_item_pair):
        alpha=0.3
        activity_set = list()
        activity_set.append(node_item_pair)
        activity_nodes = list()
        activity_nodes.append(node_item_pair)
        while activity_set:
            new_activity_set = list()
            for user, item_id in activity_set:
                for node, weight in self.graph.get_neighbors(user):
                    if (node,item_id) not in activity_nodes:
                        if random.random() < weight*self.graph.user_item_sim[node][item_id]:
                            activity_nodes.append((node,item_id))
                            new_activity_set.append((node,item_id))
                for a_item, weight in self.graph.get_item_neighbors(self.graph.item_list[item_id]):
                    a_item_id = self.graph.item_list.index(a_item)
                    if (user,a_item_id) not in activity_nodes:
                        if random.random() < alpha*self.graph.user_item_sim[user][a_item_id]:
                            activity_nodes.append((user,a_item_id))
                            new_activity_set.append((user,a_item_id))


            activity_set = new_activity_set
        return activity_nodes

class Worker(mp.Process):
    def __init__(self, inQ, outQ, func, user_num, item_num):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.user_num = user_num
        self.item_num = item_num
        self.func = func
        self.count = 0

    def run(self):
        while True:
            theta = self.inQ.get()
            # print(theta)
            for _ in range(int(theta)):
                n1 = random.randint(0, self.user_num-1)
                n2 = random.randint(0, self.item_num-1)
                rr = self.func((n1,n2))
                self.R.append(rr)
                # self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def create_worker(worker, num, model, graph):
    """
        create processes
        :param num: process number
        :param task_num: the number of tasks assigned to each worker
    """
    func = generate_rr(graph)
    if model == "LT":
        func = func.generate_rr_lt
    else:
        func = func.generate_rr_ic


    for i in range(num):
        # print(i)
        worker.append(Worker(mp.Queue(), mp.Queue(), func, graph.user_num, graph.item_num))
        worker[i].start()


def finish_worker(worker):
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()

def logcnk(n, k):
    res = 0
    for i in range(n-k+1, n+1):
        res += math.log(i)
    for i in range(1, k+1):
        res -= math.log(i)
    return res

def sampling(graph, k, epsoid, l, model):
    n = graph.user_num * graph.item_num
    R = []
    LB = 1
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = 8
    worker = []
    create_worker(worker, worker_num, model, graph)
    for i in tqdm(range(1, int(math.log2(n-1))+1)):
        s = time.time()
        x = n/(math.pow(2, i))
        lambda_p = ((2+2*epsoid_p/3)*(logcnk(n, k) + l*math.log(n) + math.log(math.log2(n)))*n)/pow(epsoid_p, 2)
        theta = lambda_p/x
        # print(theta-len(R))
        for ii in range(worker_num):
            worker[ii].inQ.put((theta-len(R))/worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list 
        # finish_worker()
        # worker = []
        end = time.time()
        print('time to find rr', end - s)
        start = time.time()
        Si, f = node_selection(R, k, n, graph.item_num)
        print(f)
        end = time.time()
        print('node selection time', time.time() - start)
        # print(F(R, Si))
        # f = F(R,Si)
        if n*f >= (1+epsoid_p)*x:
            LB = n*f/(1+epsoid_p)
            break

    # finish_worker()
    alpha = math.sqrt(l*math.log(n) + math.log(2))
    beta = math.sqrt((1-1/math.e)*(logcnk(n, k)+l*math.log(n)+math.log(2)))
    lambda_aster = 2*n*pow(((1-1/math.e)*alpha + beta), 2)*pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r
    # print(diff)
    _start = time.time()
    if diff > 0:
        # print('j')
        for ii in range(worker_num):
            worker[ii].inQ.put(diff/ worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    '''
    
    while length_r <= theta:
        v = random.randint(1, n)
        rr = generate_rr(v)
        R.append(rr)
        length_r += 1
    '''
    _end = time.time()
    print(_end - _start)
    finish_worker(worker)
    return R

def node_selection(R, k, n, item_num, mode="1"):
    Sk = set()
    rr_degree = [0 for ii in range(n)]
    node_rr_set = dict()
    # node_rr_set_copy = dict()

    matched_count = 0
    for j in range(0, len(R)):
        rr = R[j]
        for rr_node in rr:
            # print(rr_node)
            idx = rr_node[0]*item_num + rr_node[1]
            rr_degree[idx] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
                # node_rr_set_copy[rr_node] = list()
            node_rr_set[rr_node].append(j)
            # node_rr_set_copy[rr_node].append(j)
    # idx_list = list(range(n))
    user_selected = []
    item_selected = []
    while len(Sk) < k:
    # for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        user = max_point//item_num
        item = max_point%item_num

        if item not in item_selected and user not in user_selected:
            Sk.add((user,item))
            user_selected.append(user)
            item_selected.append(item)
        # Sk.add((user,item))

        matched_count += len(node_rr_set[(user,item)])
        index_set = []
        for node_rr in node_rr_set[(user,item)]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                idx = rr_node[0]*item_num + rr_node[1]
                rr_degree[idx] -= 1
                node_rr_set[rr_node].remove(jj)
        for i in range(item_num):
            idx = user * item_num + i
            rr_degree[idx] = 0

    return Sk, matched_count/len(R)


def IMM(graph, k, model='LT', epsoid=0.5, l=1):
    n = graph.user_num * graph.item_num
    l = l*(1+ math.log(2)/math.log(n))
    R = sampling(graph, k, epsoid, l, model)
    Sk, z = node_selection(R, k, n, graph.item_num)

    return Sk




def get_search_space(graph):
    search_space = []
    user_space = list(range(graph.user_num))
    item_space = list(range(graph.item_num))
    user_degree = []
    for u in user_space:
        user_degree.append(graph.get_out_degree('user',u))
    item_degree = []
    for v in item_space:
        item_degree.append(graph.get_out_degree('item',graph.item_list[v]))
    
    user_space = np.array(user_space)
    user_degree = np.array(user_degree)
    idx = np.argpartition(user_degree, -100)[-100:]
    user_space = user_space[idx].tolist()
    item_space = np.array(item_space)
    item_degree = np.array(item_degree)
    idx = np.argpartition(item_degree, -100)[-100:]
    item_space = item_space[idx].tolist()
    search_space.append(user_space)
    search_space.append(item_space)
    return search_space

def CELFPP(graph, infmodel, k):
    search_space = get_search_space(graph)
    user_space = search_space[0]
    item_space = search_space[1]
    pair2id = {}
    id2pair = {}
    S = set() # 种子集合
    influence = [] # 种子集影响力
    Q = []
    last_seed = None
    cur_best = None
    # 初始化 
    i=0
    mg1 = []
    mg2 = []
    flag = []
    prev_best = []
    maxmg1 = 0
    id2pair = []
    for u in tqdm(user_space):
        for v in item_space:
            pair2id[(u,v)] = i
            id2pair.append((u,v))            
            mg1.append(infmodel.query({(u,v)}))            
            prev_best.append(cur_best)
            if cur_best:
                mg2.append(infmodel.query({(u,v)}|{cur_best}))
            else:
                mg2.append(mg1[i])
            flag.append(0)
            Q.append((u,v))
            if mg1[i] > maxmg1:
                maxmg1 = mg1[i]
                cur_best = (u,v)
            i+=1
    item_selected = []
    user_selected = []           
    while len(S) < k:
        p = heapq.heappop(Q)
        if flag[pair2id[p]] == len(S):
            print('|S|:',len(S))
            if p[0] not in user_selected and p[1] not in item_selected:
                S.add(p)
                last_seed = p
                user_selected.append(p[0])
                item_selected.append(p[1])
                continue
            # else:
            #     continue
        elif prev_best[pair2id[p]] == last_seed:
            mg1[pair2id[p]] = mg2[pair2id[p]]  
        else:
            mg1[pair2id[p]] = infmodel.query(S|{p}) - infmodel.query(S)
            prev_best[pair2id[p]] = cur_best
            mg2[pair2id[p]] = infmodel.query(S|{p}|{cur_best}) - infmodel.query(S|{cur_best}) 
        flag[pair2id[p]] = len(S)
        cur_best = max(id2pair, key=lambda x: mg1[pair2id[x]])
        heapq.heappush(Q, p)
    return S

