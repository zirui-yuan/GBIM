import torch
import numpy as np
import time

import random
import pickle
from collections import defaultdict
import argparse

from tqdm import tqdm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import MyDataset, MyDataXset
from graph import pGraph
from model import MultiInfSurrogate
from InfModel import MultiLT, MultiIC
from utils import adj_mul

print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Main")
datasets = ['ciao', 'epinions','delicious','lastfm','synthetic']
parser.add_argument("-d", "--dataset", default="lastfm", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT']
parser.add_argument("-dm", "--diffusion_model", default="IC", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
parser.add_argument('-k', type=int, default=5, help='seed set size')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers for deep methods')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--num_heads', type=int, default=4)

parser.add_argument('--M', type=int,
                    default=20, help='number of random features')
parser.add_argument('--K', type=int, default=10, help='num of samples for gumbel softmax sampling')
parser.add_argument('--tau', type=float, default=1, help='temperature for gumbel softmax')
parser.add_argument('--rb_order', type=int, default=1, help='order for relational bias, 0 for not use')
parser.add_argument('--rb_trans', type=str, default='sigmoid', choices=['sigmoid', 'identity'],
                    help='non-linearity for relational bias')
args = parser.parse_args()
args.early_stop = 5
args.use_gumbel = False
args.use_residual = True
args.use_bn = True
args.use_act = True
args.use_bias = True
args.seed = 2023
sample_way = 'random' # initial traing data D
graphsample = 1000 # sample items 

print(args.k)
print(args.dataset)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(args.seed)

file_path = 'data/' + args.dataset + '/' + args.diffusion_model  + '_' + str(graphsample)  + '_' + str(sample_way) + '_' + str(args.k) + '.pkl'
model_path = 'model_state/'+ args.dataset + '/' + args.diffusion_model  + '_' + str(graphsample)  + '_' + str(sample_way) + '_' + str(args.k) + '.pt'
with open(file_path, 'rb') as f:
    datasets = pickle.load(f)

graph = pGraph(data_name=args.dataset, sample=graphsample)    
adj = graph.get_user_adj()
n = graph.user_num
m = graph.item_num
adj = torch.Tensor(adj.toarray()).to_sparse().indices().to(device)
### Adj storage for relational bias ###
adjs = []
adjs.append(adj)
for i in range(args.rb_order - 1): # edge_index of high order adjacency
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)

def model_test(model,test_loader):
    model.eval()
    test_loss = 0
    y_label_all = []
    y_pred_all = []
    for batch_idx, data_pair in enumerate(test_loader):
        x = data_pair[0].to(device)
        y = data_pair[1].to(device)    
        with torch.no_grad():
            y_pred = model(x, adjs, args.tau)
            loss = model.loss(y_pred,y)
        y_label_all.append(y.detach().cpu().numpy())
        y_pred_all.append(y_pred.detach().cpu().numpy())
        test_loss += loss * y_pred.shape[0]
    y_label_all = np.concatenate(y_label_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_label_threshold = np.percentile(y_label_all, 90)
    y_label_all = np.where(y_label_all >= y_label_threshold, 1, 0)
    if True in np.isnan(y_pred_all):
        auc = 0
    else:
        auc = roc_auc_score(y_label_all,y_pred_all)
    model.train()
    return test_loss/len(y_pred_all), auc

def train(datasets):
    random.shuffle(datasets)
    train_data, test_data = train_test_split(datasets, test_size=0.3)
    test_set = MyDataset(test_data)
    test_loader  = DataLoader(dataset=test_set,  batch_size=args.batch_size, shuffle=False)
    train_set = MyDataset(train_data)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = MultiInfSurrogate(n=n, item_num=m, hidden_channels=args.hidden_channels, out_channels=m, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                 nb_random_features=args.M, use_bn=args.use_bn, use_gumbel=args.use_gumbel,
                 use_residual=args.use_residual, use_act=args.use_act, use_bias = args.use_bias, nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    model = model.to(device)
    min_test_loss = float('inf')
    max_auc = 0
    early_stop_ct = 0
    for epoch in range(args.epochs):
        total_loss = 0
        y_label_all = []
        y_pred_all = []
        model.train()
        for batch_idx, data_pair in enumerate(train_loader):
            x = data_pair[0].to(device)
            y = data_pair[1].to(device)
            optimizer.zero_grad()
            y_pred = model(x, adjs, args.tau)
            loss = model.loss(y_pred,y)
            total_loss += loss * y_pred.shape[0]
            loss.backward()
            optimizer.step()
            y_label = y.detach().cpu().numpy()
            y_label_all.append(y_label)
            y_pred_np = y_pred.detach().cpu().numpy()
            y_pred_all.append(y_pred_np)
        y_label_all = np.concatenate(y_label_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_label_threshold = np.percentile(y_label_all, 90)
        y_label_all = np.where(y_label_all >= y_label_threshold, 1, 0)
        if True in np.isnan(y_pred_all):
            auc = 0
            break
        else:
            auc = roc_auc_score(y_label_all,y_pred_all)
        print("Epoch: {}".format(epoch+1), 
            "\tTrain loss: {:.4f}".format(total_loss / len(train_set)),
            "\tTrain AUC: {:.4f}".format(auc)
            )
        test_loss,auc = model_test(model,test_loader)
        if test_loss < min_test_loss:
        # if auc > max_auc:
            print( 
            "\tTest loss: {:.4f}".format(test_loss),
            "\tTest AUC: {:.4f}".format(auc)
            )
            min_test_loss = test_loss
            max_auc = auc
            early_stop_ct = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stop_ct += 1
            print( 
            "\tTest loss: {:.4f}".format(test_loss),
            "\tTest AUC: {:.4f}".format(auc),
            "\tEarly Stop Ct: {}".format(early_stop_ct)
            )
        if early_stop_ct >= args.early_stop:   
            print('early stop, min_test_loss:{} final auc: {}'.format(min_test_loss, max_auc))     
            break
    print('min_test_loss:{} final auc: {}'.format(min_test_loss, max_auc))  
    # test_loss, prec= model_test(model, last=True)
    # print('Final results: '
    #         "\tTest loss: {:.4f}".format(test_loss),
    #         "\tPrec: {:.4f}".format(prec),
    #         )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def acq_func_EI(y_max, beta, phi_x, m_T, Kinv):
    u = m_T @ phi_x.T
    u = u.cpu().numpy()
    sigma =  torch.sum((phi_x @ Kinv) * phi_x, -1) + 1/beta
    sigma = sigma.cpu().numpy()
    y_max = y_max.cpu().numpy()
    z_x = (u-y_max)/sigma
    acq_v = (u-y_max)*norm.cdf(z_x) + sigma*norm.pdf(z_x)
    return acq_v

def BayesianLRParamterCalc(datasets, model, beta, alpha):
    #construct design matrix
    print('construct design matrix...')
    design_matrix = []
    y_label = [] 
    dataset = MyDataset(datasets)
    data_loader  = DataLoader(dataset=dataset,  batch_size=1, shuffle=False)
    for data_pair in tqdm(data_loader):
        x = data_pair[0].to(device)
        y = data_pair[1].to(device)
        with torch.no_grad():
            y_pred = model.lastlayer(x, adjs, args.tau)
        design_matrix.append(y_pred)
        y_label.append(y)
    design_matrix = torch.vstack(design_matrix) #[N, d]
    y_label = torch.concatenate(y_label) #[N,]
    y_max = y_label.max()
    print('y_max:{}'.format(y_max))
    D = design_matrix.shape[1]
    K = beta * torch.matmul(design_matrix.T, design_matrix) + alpha*torch.eye(D).to(device)
    Kinv = torch.linalg.inv(K)
    m_T = beta*Kinv @ design_matrix.T @ y_label
    return y_max, m_T, Kinv

def data_x2seeds(data_x):
    indices = torch.nonzero(data_x).flatten().tolist()
    seeds = set(zip(indices, data_x[indices].tolist()))
    return seeds

def get_sample_prob_from_data(datasets):
    inf = [datasets[i][1] for i in range(len(datasets))]
    data_x = [datasets[i][0] for i in range(len(datasets))]
    inf = torch.stack(inf).numpy()
    threshold = np.percentile(inf, 97)
    selected_id = np.where(inf >= threshold)[0]
    data_x = [data_x[i] for i in selected_id]
    seeds_data = [data_x2seeds(x) for x in data_x]
    user_space = []
    item_space = []
    user_ct = defaultdict(int)
    item_ct = defaultdict(int)
    ct = 0
    for seeds in seeds_data:
        for user,item in seeds:
            user_space.append(user)
            user_ct[user] += 1
            item_space.append(item)
            item_ct[item] += 1
            ct+=1
    user_space = list(set(user_space))
    item_space = list(set(item_space))
    user_prob = [user_ct[user]/ct for user in user_space]
    item_prob = [item_ct[item]/ct for item in item_space]
    return user_space, item_space, user_prob, item_prob

def MaxEI(datasets, graph, model,k, sample_num=20000,  alpha = 2.0, beta = 25):
    y_max, m_T, Kinv = BayesianLRParamterCalc(datasets, model, beta, alpha)
    data_x = []
    seeds_data = []
    a=0.5
    user_space = list(range(graph.user_num))
    item_space = list(range(graph.item_num))
    seeds_exist = [data_x2seeds(datasets[i][0]) for i in range(len(datasets))]
    print('sample new data ...')
    for _ in tqdm(range(int(sample_num*(1-a)))):
        seeds=set()
        x = torch.zeros(graph.user_num,dtype = torch.long)
        users = random.sample(user_space, k)
        items = random.sample(item_space, k)
        for i in range(k):
            seeds.add((users[i],items[i]))
            x[users[i]] = items[i]
        if seeds in seeds_exist:
            continue
        data_x.append(x)
        seeds_data.append(seeds)
        seeds_exist.append(seeds)
    user_space, item_space, user_prob, item_prob = get_sample_prob_from_data(datasets)
    for _ in tqdm(range(int(sample_num*a))):
        seeds=set()
        x = torch.zeros(graph.user_num,dtype = torch.long)
        users = np.random.choice(a=user_space,size=k, replace=False, p=user_prob)
        items = np.random.choice(a=item_space,size=k, replace=False, p=item_prob)
        for i in range(k):
            seeds.add((users[i],items[i]))
            x[users[i]] = items[i]
        if seeds in seeds_exist:
            continue
        data_x.append(x)
        seeds_data.append(seeds)
        seeds_exist.append(seeds)
    a_datasets = MyDataXset(data_x)
    a_dataloader = DataLoader(dataset = a_datasets, batch_size=args.batch_size, shuffle=False)
    EIval = []
    print('\nEI val calc...')
    for batch_idx, x in tqdm(enumerate(a_dataloader)):
        x = x.to(device)
        with torch.no_grad():
            phi_x_batch = model.lastlayer(x, adjs, args.tau)
            EIval_batch = acq_func_EI(y_max, beta, phi_x_batch, m_T, Kinv)
            EIval.append(EIval_batch)
    EIval = np.concatenate(EIval)
    threshold = np.percentile(EIval, 99.9)
    selected_id = np.where(EIval >= threshold)[0]
    del EIval
    if len(selected_id) == sample_num:
        selected_id = random.sample(selected_id.tolist(),int(sample_num*0.001))
        selected_id = np.array(selected_id)
    seeds_data = [seeds_data[i] for i in selected_id]
    data_x = [data_x[i] for i in selected_id]
    return seeds_data,data_x

def NewdataEval(infmodel, seeds_data, data_x, datasets):
    print('\ninfmodel eval...')
    data_y = []
    for seeds in tqdm(seeds_data):
        y = torch.tensor(np.sum(infmodel.query(seeds)),dtype=torch.float32)
        data_y.append(y)
    data = list(zip(data_x, data_y))
    datasets.extend(data)
    random.shuffle(datasets)
    y = [datasets[i][1] for i in range(len(datasets))]
    y = torch.stack(y)
    idx = y.argmax()
    data_x = datasets[idx][0]
    indices = torch.nonzero(data_x).flatten().tolist()
    seeds = set(zip(indices, data_x[indices].tolist()))
    # print('current max: {}'.format(y.max()))
    return datasets, seeds

def MaxAquisitionSelection(datasets, graph, k, sample_num=20000, acq_fun = 'EI'):
    if acq_fun == 'EI':
        return MaxEI(datasets, graph, k, sample_num)

def eval_seeds(InfModel, seeds, best, i=None):
    mean, std = InfModel.query(seeds, return_std = True)
    if i:
        if mean > best:
            best = mean
        print("step {}:  {}".format(i, best))
        return mean, std, best
    else:
        print("multi-inf {}".format(mean))
        return mean, std

def add_new_data(seeds, train_data):
    init_state = torch.zeros((graph.user_num, graph.item_num))
    for user, item in seeds:
        init_state[user][item] = 1
    init_state_sparse = init_state.to_sparse()
    final_status = InfModel.query(seeds)
    final_status = torch.FloatTensor(final_status)
    train_data.append((init_state_sparse, final_status))
    random.shuffle(train_data)
    return train_data

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
    idx = np.argpartition(user_degree, -200)[-200:]
    user_space = user_space[idx].tolist()
    item_space = np.array(item_space)
    item_degree = np.array(item_degree)
    idx = np.argpartition(item_degree, -100)[-100:]
    item_space = item_space[idx].tolist()
    search_space.append(user_space)
    search_space.append(item_space)
    return search_space

def GBIM(graph, datasets, infmodel, k, OptNum = 100, acq_fun='EI'):
    start = time.time()
    results = []
    runtime = []
    best = 0
    for i in tqdm(range(OptNum)):
        model = train(datasets)
        seeds_data, data_x = MaxAquisitionSelection(datasets, graph, model, k, acq_fun=acq_fun)
        datasets, seeds = NewdataEval(infmodel, seeds_data, data_x, datasets)
        _, _, best = eval_seeds(infmodel, seeds, best, i+1)
        results.append(best)
        t=time.time()-start
        print("runtime:".format(t))
        runtime.append(t)
        log_results(results,runtime)
    return seeds

def log_results(results,runtime):
    with open('resultslog.txt', 'w+') as f:
        f.write(str(results) + '\n')
        f.write(str(runtime) + '\n')

# seeds test
# infmodel params
beta = 0.3
simulate_times = 100
if args.diffusion_model == 'IC':
    InfModel = MultiIC(graph,beta=beta,simulate_times=simulate_times)
elif args.diffusion_model == 'LT':
    InfModel = MultiLT(graph,beta=beta,simulate_times=simulate_times)

seeds = GBIM(graph = graph, datasets=datasets, infmodel=InfModel, k=args.k)
eval_seeds(InfModel, seeds)
print('done')

