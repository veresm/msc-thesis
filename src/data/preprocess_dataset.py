"""
Moves the files from the raw folder to interim while separating features, users, items
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

def preprocess(data_path, order):
  if order == 'uti':
      user_idx = 0
      ts_idx = 1
      item_idx = 2
      
  elif order == 'uit':
      user_idx = 0
      item_idx = 1
      ts_idx = 2

  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_path) as f:
    s = next(f)
    for idx, line in tqdm(enumerate(f)):
      e = line.strip().split(',')
      if e[0] == '' or e[1] == '' or e[2] == '':
          print(f"Error loading line {idx}: {line}")
          continue
      u = int(e[user_idx])
      i = int(e[item_idx])

      ts = float(e[ts_idx])
      label = float(0)
      
      feat = np.array([float(0)])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_path, file, bipartite=True):
  data_name = os.path.splitext(os.path.basename(file))[0]
  Path(f"{data_path}/{data_name}").mkdir(parents=True, exist_ok=True)
  
  OUT_DF = f"{data_path}/{data_name}/ml_{data_name}.csv"
  OUT_FEAT = f"{data_path}/{data_name}/ml_{data_name}.npy"
  OUT_NODE_FEAT = f"{data_path}/{data_name}/ml_{data_name}_node.npy"
  
  if data_name != "rtl_stream":
      order = 'uit'
  else:
      order = 'uti'

  df, feat = preprocess(file, order)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)
  
def preprocess_dataset(data='../../data', bipartite=True): 
    raw_data_path = os.path.join(data, "raw/")
    interim_data_path = os.path.join(data, "interim/")
    
    files = os.listdir(raw_data_path)
    files = [os.path.join(raw_data_path, x) for x in files if os.path.splitext(x)[1] == '.csv']
    
    for file in files:
        data_name = os.path.splitext(os.path.basename(file))[0]
        print(f"Preprocess {data_name}...")
        run(interim_data_path, file, bipartite = True)


