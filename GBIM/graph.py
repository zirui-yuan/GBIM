from tqdm import tqdm
import numpy as np
import pickle
import random
import os
from scipy.sparse import csr_matrix

class pGraph():
    """
        graph data structure to store the network
    :return:
    """
    def __init__(self, data_name = "ciao", sample = 0.2):
        self.data_name = data_name
        self.sample_rate = None
        if sample < 1:
            self.sample_rate = sample            
        else:
            self.sample_num = sample
        self.root_path = "./data/" + self.data_name
        self.network = dict()
        self.item_net = dict()
        self.load_network()
        self.item_list = list(self.item_net.keys())
        self.user_list = list(self.network.keys())
        self.user_num = len(self.network.keys())
        self.item_num = len(self.item_list)
        self.user_item_sim = self.init_user_item_sim()    

    def add_node(self, node, net):
        if node not in net:
            net[node] = dict()

    def add_edge(self, s, e, w, net):
        """
        :param s: start node
        :param e: end node
        :param w: weight
        """
        pGraph.add_node(self, s, net)
        pGraph.add_node(self, e, net)
        net[s][e] = w

    def get_out_degree(self, net, source):
        if net == 'user':
            return len(self.network[source])
        elif net == 'item':
            return len(self.item_net[source])

    def get_neighbors(self, source):
        return self.network[source].items()
    
    def get_item_neighbors(self, source):
        return self.item_net[source].items()
    
    def load_network(self):
        with open(self.root_path + "/ratings.pkl", 'rb') as fs:
            rating = pickle.load(fs)
        with open(self.root_path + "/trust.pkl", 'rb') as fs:
            trust = pickle.load(fs)
        if len(self.network) != 0:
            self.network = dict()
        src_ids = trust.tocoo().row
        tgt_ids = trust.tocoo().col
        nodes = np.unique(tgt_ids)

        for tgt_id in tqdm(nodes):
            idx = np.where(tgt_ids==tgt_id)[0]
            src = src_ids[idx]
            w = 1/len(src)
            for i, src_id in enumerate(src):
                self.add_edge(src_id, tgt_id, w, self.network)

        file_path = self.root_path + '/item_net/'+str(self.sample_num)+'.pkl'
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as fs:
                self.item_net = pickle.load(fs)
        elif self.data_name == 'synthetic':
            item_edges = np.load(self.root_path + "/item1.npy")
            for i, edges in enumerate(item_edges):
                self.add_edge(edges[0], edges[1], 1, self.item_net)
        else:
            idx = list(range(rating.shape[1]))
            if self.sample_rate:
                sample_idx = random.sample(idx, int(len(idx)*self.sample_rate))
            else:
                sample_idx = random.sample(idx, self.sample_num)
            rating = rating[:,sample_idx]
            item_sim = self.get_item_sim(rating)
            sim_val = item_sim.tocoo().data
            src_ids = item_sim.tocoo().row
            tgt_ids = item_sim.tocoo().col
            idx = np.where(sim_val>0.5)[0]
            sim_val = sim_val[idx]
            src_ids = src_ids[idx]
            tgt_ids = tgt_ids[idx]
            nodes = np.unique(tgt_ids)
            for tgt_id in tqdm(nodes):
                idx = np.where(tgt_ids==tgt_id)[0]
                src = src_ids[idx]
                w = sim_val[idx]

                idx = np.where(src!=tgt_id)[0]
                src = src[idx]
                w = w[idx]
                
                # w = torch.from_numpy(w)
                # w = F.softmax(w, 0)
                # w = w.numpy()
                for i, src_id in enumerate(src):
                    self.add_edge(src_id, tgt_id, w[i], self.item_net)
            with open(file_path, 'wb') as f:
                pickle.dump(self.item_net, f)
    
    def get_item_sim(self, rating):
        norm = np.sqrt(rating.power(2).sum(0))
        N = norm.shape[1]
        norm = np.array(norm).reshape(N,)
        rating.data = rating.data / norm[rating.indices]
        rating = rating.T * rating
        return rating
    
    def init_user_item_sim(self):
        file_path = './data/'+str(self.data_name)+'/user_item_sim.pkl'
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as fs:
                user_item_sim = pickle.load(fs)
        elif self.data_name == 'synthetic':
            user_item_sim = [[0]* self.item_num for _ in range(self.user_num)]
            for i in range(self.user_num):
                for j in range(self.item_num):
                    user_item_sim[i][j] = random.uniform(0.2, 0.8)
            with open(file_path, 'wb') as f:
                pickle.dump(user_item_sim, f)
        else:
            user_item_sim = [[0]* self.item_num for _ in range(self.user_num)]
            with open(self.root_path + "/ratings.pkl", 'rb') as fs:
                rating = pickle.load(fs)

            rating = rating[:,self.item_list]
            item_item_sim = self.get_item_sim(rating).todense()
            rating = rating.tocoo()
            max_val = rating.data.max()
            for j in tqdm(range(self.item_num)):
                item_id = self.item_list[j]
                for i in range(self.user_num):
                    idx = rating.row == i
                    item_ids = rating.col[idx]
                    labels = rating.data[idx]
                    if item_id in item_ids:
                        user_item_sim[i][j] = labels[np.where(item_ids==item_id)] / max_val
                    else:
                        b = sum([item_item_sim[j, self.item_list.index(item_ids[k])] for k in range(len(item_ids)) if item_ids[k] in self.item_list]) 
                        if b==0:
                            user_item_sim[i][j] = random.uniform(0.2, 0.8)
                        else:
                            user_item_sim[i][j] = sum([item_item_sim[j, self.item_list.index(item_ids[k])]*labels[k] for k in range(len(item_ids)) if item_ids[k] in self.item_list]) / b                       
                            user_item_sim[i][j] /= max_val                        
            with open(file_path, 'wb') as f:
                pickle.dump(user_item_sim, f)
        return user_item_sim
    
    def get_user_adj(self):
        rows = []
        cols = []
        val = []
        size = len(self.network.keys())
        for i in self.network.keys():
            for j in self.network[i].keys():
                rows.append(i)
                cols.append(j)
                # val.append(self.network[i][j])
                val.append(1)
        # adj = SparseTensor(row=torch.LongTensor(rows),col=torch.LongTensor(cols),value = torch.tensor(val),sparse_sizes=(size,size))
        adj = csr_matrix((val,(rows,cols)))
        return adj