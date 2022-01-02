import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import csv
from datetime import datetime
#%%

def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []
  idx_file=0
  path = "../data/temp.csv"
  csv_file = open(path, "w", newline='')
  writer = csv.writer(csv_file)

  with open(data_name) as f:
    s = next(f)
    for idx, line in tqdm(enumerate(f)):
      e = line.strip().split(',')
      u = int(float(e[0]))
      i = int(float(e[1]))

      ts = float(e[2])
      label = int(0)
      feat = int(0)
      
      writer.writerow([u,i,ts,label,idx])
      
  return path


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    if not (df.u.max() - df.u.min() + 1 == len(df.u.unique())):
        print("Relabeling users...")
        new_df.u = pd.factorize(new_df.u.tolist())[0]
        # mapper = {original:new for new, original in enumerate(df.u.unique())}
        # for idx, row in tqdm(new_df.iterrows()):
        #     new_df.loc[idx,'u'] = mapper[row.u]
    if not (df.i.max() - df.i.min() + 1 == len(df.i.unique())):
        print("RElabeling items...")
        new_df.i = pd.factorize(new_df.i.tolist())[0]
        # mapper = {original:new for new, original in enumerate(df.i.unique())}
        # for idx, row in tqdm(new_df.iterrows()):
        #     new_df.loc[idx,'i'] = mapper[row.i]
            

    upper_u =new_df.u.max() + 1
    new_i = new_df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True):
  Path("../data/").mkdir(parents=True, exist_ok=True)
  PATH = '../data/{}.csv'.format(data_name)
  OUT_DF = '../data/ml_{}.csv'.format(data_name)
  OUT_FEAT = '../data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = '../data/ml_{}_node.npy'.format(data_name)

  ##############
  path = preprocess(PATH)
  df = pd.read_csv(path,
                   names=['u', 'i', 'ts','label','idx'])
  df.fillna(0, inplace=True)
  df.label = df.label.astype(int)
  df.idx = df.index
  feat = np.array([0]*df.shape[0])
  print("Start reindexing")
  ##################
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)
#%%
# parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
# parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
#                     default='rtl_vv')
# parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

# args = parser.parse_args()

# run(args.data, bipartite=args.bipartite)

# run("rtl_vv_2021", bipartite=True)

df = pd.read_csv("../data/temp.csv", names=['u', 'i', 'ts','label','idx'])
print("data loaded")
df.fillna(0, inplace=True)
df.label = df.label.astype(int)
df.idx = df.index
feat = np.array([0]*df.shape[0])
feat = np.expand_dims(feat, 1)
print("Strat reindexing")

new_df = reindex(df, bipartite=True)
print("Done reindexing!")
#%%
min_ts = new_df.ts.min() + 1
new_df.ts = new_df.ts - min_ts
#%%
empty = np.zeros(feat.shape[1])[np.newaxis, :]
feat = np.vstack([empty, feat])

max_idx = max(new_df.u.max(), new_df.i.max())
rand_feat = np.zeros((max_idx + 1, 172))

data_name = "rtl_vv_2021"

print("Start saving")

OUT_DF = '../data/ml_{}.csv'.format(data_name)
OUT_FEAT = '../data/ml_{}.npy'.format(data_name)
OUT_NODE_FEAT = '../data/ml_{}_node.npy'.format(data_name)
new_df.to_csv(OUT_DF)
np.save(OUT_FEAT, feat)
np.save(OUT_NODE_FEAT, rand_feat)
