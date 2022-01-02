import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import networkx as nx
import json
import random


def save_gml(train_set, path, data_name):
    gml_save_path = f"{path}{data_name}_train.gml"
    label_save_path = f"{path}{data_name}_train.json"
    
    G = nx.Graph()
    G.add_nodes_from(set(train_set.u.tolist()), bipartite=0)
    G.add_nodes_from(set(train_set.i.tolist()), bipartite=1)
    G.add_edges_from(train_set[['u', 'i']].values)

    nx.write_gml(G, gml_save_path)
    labels = {}
    for idx, l in enumerate(G.nodes):
        labels[l] = idx
    
    with open(label_save_path, 'w') as fp:
        json.dump(labels, fp)
        
def negative_sample(ml_data, ml_data_test):
    n_samp = 0
    p_samp = ml_data_test.shape[0]
    sampling_factor = 1
    while n_samp/p_samp <= 1:
        sampling_factor += 0.25
        ml_data_test_true = ml_data_test.copy(deep=True)
        
        ml_data_test_false = ml_data_test_true.sample(n = int(p_samp*sampling_factor), replace=True)
        ml_data_test_false.u = np.random.permutation(ml_data_test_false.u)
        
        ml_data_test_false = ml_data_test_false.drop_duplicates()
                
        on=['u', 'i']
        
        ml_data_test_false = (ml_data_test_false.merge(ml_data[on],
                                                       on=on,
                                                       how='left',
                                                       indicator=True)
                                  .query('_merge == "left_only"')
                                  .drop('_merge', 1))
    
        ml_data_test_false = (ml_data_test_false.merge(ml_data_test_true[on],
                                                   on=on,
                                                   how='left',
                                                   indicator=True)
                              .query('_merge == "left_only"')
                              .drop('_merge', 1))
        
        ml_data_test_true['ground_truth'] = [1]*ml_data_test_true.shape[0]
        ml_data_test_false['ground_truth'] = [0]*ml_data_test_false.shape[0]

        n_samp = ml_data_test_false.shape[0]
        print("\tRatio of negative samples: ",n_samp/p_samp)
        
    ml_data_test_false = ml_data_test_false.sample(n = p_samp, replace = False)
    ml_data_test_false.sort_values(by='ts', ascending=True, inplace=True)
    
    ml_data_test_false['idx'] = ml_data_test_true['idx'].values
    ml_data_test_false['ts'] = ml_data_test_true['ts'].values
    
    test_set = pd.concat([ml_data_test_true, ml_data_test_false])
    
    return test_set, ml_data_test_true, ml_data_test_false
        
def edge_split(ml_data, ml_data_feat,test_size=0.2,random_state=111):
    test_time = list(np.quantile(ml_data.ts, [1-test_size]))
    
    sources = ml_data.u.values
    destinations = ml_data.i.values
    edge_idxs = ml_data.idx.values
    labels = ml_data.label.values
    timestamps = ml_data.ts.values
    
    random.seed(random_state)
    
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)
    
    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps >= test_time]).union(
        set(destinations[timestamps >= test_time]))
    
    train_node_set = set(sources[timestamps < test_time]).union(
        set(destinations[timestamps < test_time]))
    
    new_test_node_set = test_node_set.difference(train_node_set)
    new_test_source_nodes = ml_data[timestamps >= test_time].u.apply(lambda x: x in new_test_node_set)
    new_test_destination_nodes = ml_data[timestamps >= test_time].i.apply(lambda x: x in new_test_node_set)
    
    new_test_edges_mask = np.logical_or(new_test_source_nodes, new_test_destination_nodes)
    
    new_test_edges_ratio = sum(new_test_edges_mask) / new_test_edges_mask.size
    if new_test_edges_ratio >= 0.25:
        print(f"!!! The number of new edges in the dataset is too high. The ratio is {new_test_edges_ratio}")
    
    new_only_test_nodes = set()
    if new_test_edges_ratio < 0.1:
        # need more new edges
        old_nodes_in_test = test_node_set & train_node_set
        
        while new_test_edges_ratio < 0.1:
            nodes = random.sample(old_nodes_in_test, int(len(old_nodes_in_test)*0.025))
            # print(f"{len(nodes)} \n", end="\r")
            # test_node_set = test_node_set | set(nodes)
            new_only_test_nodes = new_only_test_nodes | set(nodes)
            
            new_test_node_set =  test_node_set.difference(train_node_set)
            new_test_node_set = new_test_node_set | new_only_test_nodes
            
            new_test_source_nodes = ml_data[timestamps >= test_time].u.apply(lambda x: x in new_test_node_set)
            new_test_destination_nodes = ml_data[timestamps >= test_time].i.apply(lambda x: x in new_test_node_set)
            
            new_test_edges_mask = np.logical_or(new_test_source_nodes, new_test_destination_nodes)
            
            new_test_edges_ratio = sum(new_test_edges_mask) / new_test_edges_mask.size
            print(f"\tRatio of new edges in the test set: {sum(new_test_edges_mask)} / {new_test_edges_mask.size}", end="\r")
    
    # mask for new_test_node_set in train
    print(f"\tRatio of new edges in the test set: {new_test_edges_ratio}")
    
    train_mask = np.logical_and(timestamps < test_time,
                                np.logical_and(ml_data.u.apply(lambda x: x not in new_only_test_nodes),
                                               ml_data.i.apply(lambda x: x not in new_only_test_nodes)))
    ml_data_train, ml_data_feat_train = ml_data[train_mask], ml_data_feat[train_mask]
    # mask for test
    test_mask = (timestamps >= test_time)
    ml_data_test, ml_data_feat_test = ml_data[test_mask], ml_data_feat[test_mask]
    ml_data_test, ml_data_test_true, ml_data_test_false = negative_sample(ml_data, ml_data_test)

    return ml_data_train, (ml_data_test, ml_data_test_true, ml_data_test_false), ml_data_feat_train, ml_data_feat_test

def time_split_dataset(path, data_name, processed_data_path, intervals=10):
    ml_data = pd.read_csv(os.path.join(path, f"ml_{data_name}.csv"), index_col=0)
    ml_data_feat = np.load(os.path.join(path, f"ml_{data_name}.npy"))
    ml_data_node = np.load(os.path.join(path, f"ml_{data_name}_node.npy"))
    
    for interval in range(intervals):
        n_edge = int(ml_data.shape[0] * ((interval+1)/intervals))
        ml_data_interval = ml_data.iloc[:n_edge]
        ml_data_feat_interval = ml_data_feat[:n_edge+1]
        ml_data_node_interval = ml_data_node[:n_edge]
        
        Path(f"{processed_data_path}/{data_name}/{data_name}_{interval}").mkdir(parents=True, exist_ok=True)
        Path(f"{processed_data_path}/{data_name}/{data_name}_{interval}/train").mkdir(parents=True, exist_ok=True)
        Path(f"{processed_data_path}/{data_name}/{data_name}_{interval}/test").mkdir(parents=True, exist_ok=True)
        
        TRAIN_OUT_DF = f"{processed_data_path}/{data_name}/{data_name}_{interval}/train/ml_{data_name}.csv"
        TRAIN_OUT_FEAT = f"{processed_data_path}/{data_name}/{data_name}_{interval}/train/ml_{data_name}.npy"
        TRAIN_OUT_NODE_FEAT = f"{processed_data_path}/{data_name}/{data_name}_{interval}/train/ml_{data_name}_node.npy"
        
        TEST_OUT_DF_TRUE = f"{processed_data_path}/{data_name}/{data_name}_{interval}/test/ml_{data_name}_true.csv"
        TEST_OUT_DF_FALSE = f"{processed_data_path}/{data_name}/{data_name}_{interval}/test/ml_{data_name}_false.csv"
        TEST_OUT_DF = f"{processed_data_path}/{data_name}/{data_name}_{interval}/test/ml_{data_name}.csv"
        TEST_OUT_FEAT = f"{processed_data_path}/{data_name}/{data_name}_{interval}/test/ml_{data_name}.npy"
        TEST_OUT_NODE_FEAT = f"{processed_data_path}/{data_name}/{data_name}_{interval}/test/ml_{data_name}_node.npy"
        
        feat_0 = ml_data_feat_interval[0]
        ml_data_feat_interval = ml_data_feat_interval[1:]
        
        ml_data_train, (ml_data_test, ml_data_test_true, ml_data_test_false), ml_data_feat_train, ml_data_feat_test = \
            edge_split(ml_data_interval, ml_data_feat_interval,test_size=0.2,random_state=111)
            
        train_feat = np.vstack([feat_0, ml_data_feat_train])
        ml_data_train.to_csv(TRAIN_OUT_DF)
        
        save_gml(ml_data_train, f"{processed_data_path}/{data_name}/{data_name}_{interval}/train/", data_name)
        
        np.save(TRAIN_OUT_FEAT, train_feat)
        np.save(TRAIN_OUT_NODE_FEAT, ml_data_node_interval)
        
        ml_data_test.to_csv(TEST_OUT_DF)
        ml_data_test_true.to_csv(TEST_OUT_DF_TRUE)
        ml_data_test_false.to_csv(TEST_OUT_DF_FALSE)
    
        test_feat = np.vstack([feat_0, ml_data_feat_test])
        np.save(TEST_OUT_FEAT, test_feat)
        np.save(TEST_OUT_NODE_FEAT, ml_data_node_interval)
    

def split_dataset(path, data_name, processed_data_path):
    ml_data = pd.read_csv(os.path.join(path, f"ml_{data_name}.csv"), index_col=0)
    ml_data_feat = np.load(os.path.join(path, f"ml_{data_name}.npy"))
    ml_data_node = np.load(os.path.join(path, f"ml_{data_name}_node.npy"))
    
    Path(f"{processed_data_path}/{data_name}").mkdir(parents=True, exist_ok=True)
    Path(f"{processed_data_path}/{data_name}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{processed_data_path}/{data_name}/test").mkdir(parents=True, exist_ok=True)
    
    TRAIN_OUT_DF = f"{processed_data_path}/{data_name}/train/ml_{data_name}.csv"
    TRAIN_OUT_FEAT = f"{processed_data_path}/{data_name}/train/ml_{data_name}.npy"
    TRAIN_OUT_NODE_FEAT = f"{processed_data_path}/{data_name}/train/ml_{data_name}_node.npy"
    
    TEST_OUT_DF_TRUE = f"{processed_data_path}/{data_name}/test/ml_{data_name}_true.csv"
    TEST_OUT_DF_FALSE = f"{processed_data_path}/{data_name}/test/ml_{data_name}_false.csv"
    TEST_OUT_DF = f"{processed_data_path}/{data_name}/test/ml_{data_name}.csv"
    TEST_OUT_FEAT = f"{processed_data_path}/{data_name}/test/ml_{data_name}.npy"
    TEST_OUT_NODE_FEAT = f"{processed_data_path}/{data_name}/test/ml_{data_name}_node.npy"
    
    feat_0 = ml_data_feat[0]
    ml_data_feat = ml_data_feat[1:]
    
    ml_data_train, (ml_data_test, ml_data_test_true, ml_data_test_false), ml_data_feat_train, ml_data_feat_test = \
        edge_split(ml_data, ml_data_feat,test_size=0.2,random_state=111)
        
    train_feat = np.vstack([feat_0, ml_data_feat_train])
    ml_data_train.to_csv(TRAIN_OUT_DF)
    
    save_gml(ml_data_train, f"{processed_data_path}/{data_name}/train/", data_name)
    
    np.save(TRAIN_OUT_FEAT, train_feat)
    np.save(TRAIN_OUT_NODE_FEAT, ml_data_node)
    
    ml_data_test.to_csv(TEST_OUT_DF)
    ml_data_test_true.to_csv(TEST_OUT_DF_TRUE)
    ml_data_test_false.to_csv(TEST_OUT_DF_FALSE)

    test_feat = np.vstack([feat_0, ml_data_feat_test])
    np.save(TEST_OUT_FEAT, test_feat)
    np.save(TEST_OUT_NODE_FEAT, ml_data_node)
    
def train_test_split(data='../../data'):
    interim_data_path = os.path.join(data, "interim/")
    processed_data_path = os.path.join(data, "processed/")
    
    folders = os.listdir(interim_data_path)
    for folder in folders:
        if os.path.isdir(os.path.join(interim_data_path, folder)):
            path = os.path.join(interim_data_path, folder)
            data_name = folder
            print(f"Split {data_name}...")
            split_dataset(path, data_name, processed_data_path)
            
def time_split(data='../../data', intervals=10):
    interim_data_path = os.path.join(data, "interim/")
    processed_data_path = os.path.join(data, "processed/split_data/")
    
    folders = os.listdir(interim_data_path)
    for folder in folders:
        if os.path.isdir(os.path.join(interim_data_path, folder)):
            path = os.path.join(interim_data_path, folder)
            data_name = folder
            print(f"Split {data_name} by timestamps...")
            time_split_dataset(path, data_name, processed_data_path, intervals=intervals)
            
if __name__ == "__main__":
    train_test_split()