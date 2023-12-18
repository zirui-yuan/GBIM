from responses import activate
from graph import pGraph
import random
import multiprocessing as mp
import time
import getopt
import sys
import copy
import numpy as np
from tqdm import tqdm

class InfWorker(mp.Process):
    def __init__(self, outQ, count, inf_function, seeds, user_num, item_num):
        super(InfWorker, self).__init__(target=self.start)
        self.outQ = outQ
        self.count = count
        self.inf_function = inf_function
        self.results = []
        self.seeds = seeds

    def run(self):
        # for _ in tqdm(range(self.count)):
        for _ in range(self.count):
            activate_set = self.inf_function(self.seeds, return_nodes=True)
            self.results.append(len(activate_set))

        self.outQ.put(self.results)


def create_worker(worker, num, task_num, inf_function, seeds, user_num, item_num):
    """
        create processes
        :param num: process number
        :param task_num: the number of tasks assigned to each worker
    """
    for i in range(num):
        worker.append(InfWorker(mp.Queue(), task_num, inf_function, seeds, user_num, item_num))
        worker[i].start()

def finish_worker(worker):
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()


class MultiLT:
    def __init__(self, graph, simulate_times = 1, worker_num = 8, price = None, beta = 0.3, trs_coef = 1, normal_threshold = False, normal_scale = 0.05) -> None:
        self.graph = graph
        self.threshold = self._init_threshold()
        self.threshold = trs_coef * self.threshold
        self.user_item_sim = self.graph.user_item_sim
        self.user_num = self.graph.user_num
        self.item_num = self.graph.item_num
        self.id2item = self.graph.item_list
        self.item_price = self._init_item_price(price)
        self.worker_num = worker_num
        self.simulate_times = simulate_times
        self.beta = beta
        self.normal_threshold = normal_threshold
        self.normal_scale = normal_scale

    def _init_threshold(self):
        user_item_sim = self.graph.user_item_sim
        user_item_sim = np.array(user_item_sim)
        return 1 - user_item_sim
    
    def _normal_threshold(self, sim):
        m = 1-sim
        t = np.random.normal(loc = m , scale= 0.05)
        return t
    
    def _init_item_price(self, price):
        if price == None:
            return np.ones(self.item_num)
        else:
            return np.random.rand(self.item_num)
    
    def _init_item_adj(self, hop):
        if hop > 1:
            adj = self.graph.get_item_adj().todense().getA()
            tmp = adj.copy()
            for _ in range(hop-1):                
                tmp = tmp.dot(adj)
            adj = tmp
        else:
            adj = self.graph.get_item_adj().todense().getA()
        return adj
    
    def calc_inf(self, seeds, hop = None, a_hop = None, a_m = None, return_nodes = False):
        '''
        a_hop: association hop
        a_m: max association number
        '''
        # seeds = {...,(pi,vi),...}
        now_activated_set = copy.deepcopy(seeds)
        all_activated_set = copy.deepcopy(seeds)
        node_weights = np.zeros((self.user_num, self.item_num))
        if self.normal_threshold:
            threshold = dict()
            for user in self.graph.user_list:
                threshold[user] = dict()            
        else:
            threshold = self.threshold
        inf = 0
        while now_activated_set:
            new_activated_set = set()
            # association diffusion
            now_association_set = copy.deepcopy(now_activated_set)
            while now_association_set:
                new_association_set = set()
                for user, item_id in now_association_set:
                    a_items = self.graph.get_item_neighbors(self.id2item[item_id])
                    if a_m:
                        if len(a_items) > a_m:
                            a_items = random.sample(a_items, a_m)                        
                    for a_item, weight in a_items:
                        a_item_id = self.id2item.index(a_item)
                        if (user, a_item_id) not in all_activated_set:
                            if self.normal_threshold:
                                if a_item_id in threshold[user]:
                                    sim = 1 - threshold[user][a_item_id] 
                                else:
                                    rand_t =  np.random.normal(loc = self.threshold[user][a_item_id], scale=self.normal_scale)
                                    if rand_t > 1:
                                        rand_t = 1
                                    elif rand_t < 0:
                                        rand_t = 0
                                    threshold[user][a_item_id] = rand_t
                                    sim = 1 - threshold[user][a_item_id]
                                    
                            else:
                                sim = self.user_item_sim[user][a_item_id]
                            if random.random() < self.beta*sim:
                                all_activated_set.add((user, a_item_id))
                                now_activated_set.add((user, a_item_id))
                                new_association_set.add((user, a_item_id))
                now_association_set = copy.deepcopy(new_association_set)
                if a_hop:
                    a_hop = a_hop - 1
                    if a_hop==0:
                        break 
            for seed_user, item_id in now_activated_set:
                for user, weight in self.graph.get_neighbors(seed_user):
                    if (user, item_id) not in all_activated_set:
                        node_weights[user][item_id] += weight
                        if self.normal_threshold:
                            if item_id not in threshold[user]:
                                rand_t = np.random.normal(loc = self.threshold[user][item_id] , scale = self.normal_scale)
                                if rand_t > 1:
                                    rand_t = 1
                                elif rand_t < 0:
                                    rand_t = 0
                                threshold[user][item_id] = rand_t
                        if node_weights[user][item_id] > threshold[user][item_id]:
                            all_activated_set.add((user, item_id))
                            new_activated_set.add((user, item_id))
                      
            
            now_activated_set = copy.deepcopy(new_activated_set)
            if hop:
                hop = hop - 1
                if hop == 0:
                    break
        if return_nodes:
            return all_activated_set
        for _,item_id in all_activated_set:
            inf += self.item_price[item_id]
        return inf  
          
    def query(self, seeds, return_std = False):
        worker = []
        # status = np.zeros((self.user_num, self.item_num))
        results = []
        create_worker(worker, self.worker_num, int(self.simulate_times / self.worker_num), self.calc_inf, seeds, self.user_num, self.item_num)
        for w in worker:
            # status += w.outQ.get()
            results.extend(w.outQ.get())
        finish_worker(worker)
        results = np.array(results)
        if return_std:
            return np.mean(results), np.std(results)
        
        return np.mean(results)         

class MultiIC:
    def __init__(self, graph, simulate_times = 10000, worker_num = 8, price = None, beta = 0.3, w_coef=1) -> None:
        self.graph = graph
        self.user_item_sim = self.graph.user_item_sim
        self.user_num = self.graph.user_num
        self.item_num = self.graph.item_num
        self.id2item = self.graph.item_list
        self.item_price = self._init_item_price(price)
        self.worker_num = worker_num
        self.w_coef = w_coef
        self.simulate_times = simulate_times
        self.beta = beta
        
    def _init_item_price(self, price):
        if price == None:
            return np.ones(self.item_num)
        else:
            return np.random.rand(self.item_num)
       
    def calc_inf(self, seeds, hop = None, a_hop = 1, a_m = None, return_nodes = False):
        '''
        a_hop: association hop
        a_m
        '''
        # seeds = {...,(pi,vi),...}
        now_activated_set = copy.deepcopy(seeds)
        all_activated_set = copy.deepcopy(seeds)
        inf = 0
        while now_activated_set:
            # association diffusion
            now_association_set = copy.deepcopy(now_activated_set)
            while now_association_set:
                new_association_set = set()
                for user, item_id in now_association_set:
                    a_items = self.graph.get_item_neighbors(self.id2item[item_id])
                    if a_m:
                        if len(a_items) > a_m:
                            a_items = random.sample(a_items, a_m)                        
                    for a_item, weight in a_items:
                        a_item_id = self.id2item.index(a_item)
                        if (user, a_item_id) not in all_activated_set:
                            if random.random() < self.beta*self.user_item_sim[user][a_item_id]:
                                all_activated_set.add((user, a_item_id))
                                now_activated_set.add((user, a_item_id))
                                new_association_set.add((user, a_item_id))
                now_association_set = copy.deepcopy(new_association_set)
                if a_hop:
                    a_hop = a_hop - 1
                    if a_hop==0:
                        break
            new_activated_set = set()
            for seed_user, item_id in now_activated_set:
                for user, weight in self.graph.get_neighbors(seed_user):
                    if (user, item_id) not in all_activated_set:
                        if random.random() < self.w_coef*weight*self.user_item_sim[user][item_id]:
                            all_activated_set.add((user, item_id))
                            new_activated_set.add((user, item_id))
                       
            
            now_activated_set = copy.deepcopy(new_activated_set)
            if hop:
                hop = hop - 1
                if hop == 0:
                    break
        if return_nodes:
            return all_activated_set
        for _,item_id in all_activated_set:
            inf += self.item_price[item_id]
        return inf  

    def query(self, seeds, return_std = False):
        worker = []
        results = []
        create_worker(worker, self.worker_num, int(self.simulate_times / self.worker_num), self.calc_inf, seeds, self.user_num, self.item_num)    
        for w in worker:
            results.extend(w.outQ.get())
        finish_worker(worker)
        results = np.array(results)
        if return_std:
            return np.mean(results), np.std(results)
        
        return np.mean(results) 
