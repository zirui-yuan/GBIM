import scipy.io as scio
import numpy as np
import pickle
from tqdm import tqdm
from scipy.sparse import csr_matrix

data = 'synthetic'
root_path = "data/" + data

# load data
if data == 'ciao' or data == 'epinions': 
  rating = scio.loadmat(root_path + '/rating.mat')
  trust = scio.loadmat(root_path + '/trustnetwork.mat')
  rating = rating['rating']
  trust = trust['trustnetwork']

  # undirect
  trust = np.vstack((trust, trust[:,[1,0]]))
  rating = rating[:, [0, 1, 3]]
elif data == 'delicious':
  trust = []
  with open(root_path+'/user_contacts-timestamps.dat', 'rb') as f:
      for line in f.readlines()[1:]:
          # trust.append(line)
          # d = line
          d = line.split(b'\t')
          trust.append([int(d[0]),int(d[1])])
  trust = np.array(trust)
  rating = []
  with open(root_path+'/user_taggedbookmarks-timestamps.dat', 'rb') as f:
      for line in f.readlines()[1:]:
          # trust.append(line)
          # d = line
          d = line.split(b'\t')
          rating.append([int(d[0]),int(d[1]),int(1)])
  rating = np.array(rating)
elif data == 'lastfm':
  trust = []
  with open(root_path+'/user_friends.dat', 'rb') as f:
      for line in f.readlines()[1:]:
          # trust.append(line)
          # d = line
          d = line.split(b'\t')
          trust.append([int(d[0]),int(d[1])])
  trust = np.array(trust)
  rating = []
  with open(root_path+'/user_artists.dat', 'rb') as f:
      for line in f.readlines()[1:]:
          # trust.append(line)
          # d = line
          d = line.split(b'\t')
          rating.append([int(d[0]),int(d[1]),int(1)])
  rating = np.array(rating)
elif data == 'synthetic':
   trust = np.load(root_path + "/syn05.npy")
   rating = np.load(root_path + "/ratings.npy")


trust = np.unique(trust, axis = 0)
#delete self-loop
delete = []
for i in range(len(trust)):
  if trust[i,0] == trust[i, 1]:
    delete.append(i)
trust = np.delete(trust, delete, 0)


_, index = np.unique(rating[:,0:2], axis=0, return_index=True)
rating = rating[index]

# build user index
dic_user_o2i = {}
i = 0
for user in trust.reshape(-1,):
  if user in dic_user_o2i.keys():
    continue
  else:
    dic_user_o2i[user] = i
    i += 1
print('nodes_num',len(dic_user_o2i))
print('rating_num before', len(rating))
delete = []
for i in tqdm(range(len(rating))):
  if rating[i,0] not in dic_user_o2i.keys():
    delete.append(i)
  elif rating[i,2] == 0:
    delete.append(i)
rating = np.delete(rating, delete, 0)
print('rating_num:',len(rating))

# build user index
dic_item_o2i = {}
i = 0
for item in rating[:, 1]:
  if item in dic_item_o2i.keys():
    continue
  else:
    dic_item_o2i[item] = i
    i += 1
print('items_num ', len(dic_item_o2i.keys()))

rating_x = [dic_user_o2i[u] for u in rating[:,0]]
rating_y = [dic_item_o2i[u] for u in rating[:,1]]
rating_data = csr_matrix((rating[:,2], (rating_x, rating_y)))

trust_x = [dic_user_o2i[u] for u in trust[:,0]]
trust_y = [dic_user_o2i[u] for u in trust[:,1]]
trust_data = csr_matrix((np.ones(len(trust)),(trust_x,trust_y)))

with open(root_path + "/ratings.pkl", 'wb') as f:
    pickle.dump(rating_data, f)
with open(root_path + "/trust.pkl", 'wb') as f:
    pickle.dump(trust_data, f)

print('done')