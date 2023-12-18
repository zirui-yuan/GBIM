import torch
import time
import pickle
import argparse
from tqdm import tqdm


from graph import pGraph
from InfModel import MultiLT, MultiIC
from Seeds import random_seeds, maxdegree 

print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Main")
datasets = ['ciao', 'epinions', 'delicious','lastfm','synthetic']
parser.add_argument("-d", "--dataset", default="epinions", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT']
parser.add_argument("-dm", "--diffusion_model", default="IC", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
parser.add_argument('-k', type=int, default=5, help='seed set size')
args = parser.parse_args()
k = args.k
start = time.time()
graphsample=1000
graph = pGraph(data_name=args.dataset, sample=graphsample)
beta = 0.3
sample_num = 1000
sample_way = 'random'  
simulate_times = 10
# seed_rate = 0.01L
save_path = 'data/' + args.dataset + '/' + args.diffusion_model  + '_' + str(graphsample)  + '_' + str(sample_way) + '_' + str(k) + '.pkl'
if args.diffusion_model == 'IC':
    model = MultiIC(graph,beta=beta,simulate_times=simulate_times)
elif args.diffusion_model == 'LT':
    model = MultiLT(graph,beta=beta,simulate_times=simulate_times)
print('start sampling data ...')
data = []
if sample_way == 'random':
    for i in tqdm(range(sample_num)):
        if i==0:
            seeds = set(maxdegree(graph,k))
        else:
            seeds = random_seeds(graph, k)
        x = torch.zeros(graph.user_num,dtype = torch.long)
        for user, item in seeds:
            x[user] = item
        y = torch.tensor(model.query(seeds),dtype=torch.float32)
        data.append((x, y))


print('start saving ...')
with open(save_path, 'wb') as f:
    pickle.dump(data, f)

print('done')






