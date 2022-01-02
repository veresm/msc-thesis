import numpy as np
import random
import pandas as pd
import os


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  random.seed(2020)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, path, different_new_nodes_between_val_and_test=False, randomize_features=False):
  if "_" in dataset_name:
    ### Load data and train val test split
    data_train_path = os.path.join(path, f"train/")
    data_test_path = os.path.join(path, f"test/")
    dataset_name = dataset_name.split('_')[0]
    data_full_path = os.path.join(path.split("/")[0], path.split("/")[1], f"interim/{dataset_name}")

  else:
    ### Load data and train val test split
    data_train_path = os.path.join(path, f"processed/{dataset_name}/train/")
    data_test_path = os.path.join(path, f"processed/{dataset_name}/test/")
    data_full_path = os.path.join(path, f"interim/{dataset_name}")

  # load train
  train_graph_df = pd.read_csv(os.path.join(data_train_path, f"ml_{dataset_name}.csv"))
  train_edge_features = np.load(os.path.join(data_train_path, f"ml_{dataset_name}.npy"))
  train_node_features = np.load(os.path.join(data_train_path, f"ml_{dataset_name}_node.npy"))
  
  # load_test
  test_graph_df_true = pd.read_csv(os.path.join(data_test_path, f"ml_{dataset_name}_true.csv"))
  test_graph_df_false = pd.read_csv(os.path.join(data_test_path, f"ml_{dataset_name}_false.csv"))
  test_edge_features = np.load(os.path.join(data_test_path, f"ml_{dataset_name}.npy"))
  test_node_features = np.load(os.path.join(data_test_path, f"ml_{dataset_name}_node.npy"))
  
  # load_full
  full_graph_df = pd.read_csv(os.path.join(data_full_path, f"ml_{dataset_name}.csv"))
  full_edge_features = np.load(os.path.join(data_full_path, f"ml_{dataset_name}.npy"))
  full_node_features = np.load(os.path.join(data_full_path, f"ml_{dataset_name}_node.npy"))
  
  full_sources = full_graph_df.u.values
  full_destinations = full_graph_df.i.values
  full_edge_idxs = full_graph_df.idx.values
  full_labels = full_graph_df.label.values
  full_timestamps = full_graph_df.ts.values

  full_data = Data(full_sources,
                   full_destinations,
                   full_timestamps,
                   full_edge_idxs,
                   full_labels)
  
    
  if randomize_features:
    train_node_features = np.random.rand(train_node_features.shape[0], train_node_features.shape[1])

  random.seed(2020)

  node_set = set(full_sources) | set(full_destinations)
  n_total_unique_nodes = len(node_set)

  test_sources = test_graph_df_true.u.values
  test_destinations = test_graph_df_true.i.values
  test_edge_idxs = test_graph_df_true.idx.values
  test_labels = test_graph_df_true.label.values
  test_timestamps = test_graph_df_true.ts.values
  
  test_data_true = Data(test_sources,
                   test_destinations,
                   test_timestamps,
                   test_edge_idxs,
                   test_labels)
  
  test_sources_false = test_graph_df_false.u.values
  test_destinations_false = test_graph_df_false.i.values
  test_edge_idxs_false = test_graph_df_false.idx.values
  test_labels_false = test_graph_df_false.label.values
  test_timestamps_false = test_graph_df_false.ts.values
  
  test_data_false = Data(test_sources_false,
                   test_destinations_false,
                   test_timestamps_false,
                   test_edge_idxs_false,
                   test_labels_false)
    
  # Compute nodes which appear at test time
  test_node_set = set(test_sources).union(set(test_destinations))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = test_node_set
  
  val_time = np.quantile(train_graph_df.ts, 0.85)
  
  train_sources = train_graph_df.u.values
  train_destinations = train_graph_df.i.values
  train_edge_idxs = train_graph_df.idx.values
  train_labels = train_graph_df.label.values
  train_timestamps = train_graph_df.ts.values

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = (train_timestamps <= val_time)

  train_data = Data(train_sources[train_mask],
                    train_destinations[train_mask],
                    train_timestamps[train_mask],
                    train_edge_idxs[train_mask],
                    train_labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources) | set(train_data.destinations)
  new_node_set = node_set - train_node_set

  val_mask = train_timestamps > val_time


  edge_contains_new_node_mask = np.array(
    [(a in new_node_set or b in new_node_set) for a, b in zip(train_sources, train_destinations)])
  new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)

  # validation and test with all edges
  val_data = Data(train_sources[val_mask], train_destinations[val_mask], train_timestamps[val_mask],
                  train_edge_idxs[val_mask], train_labels[val_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(train_sources[new_node_val_mask],
                           train_destinations[new_node_val_mask],
                           train_timestamps[new_node_val_mask],
                           train_edge_idxs[new_node_val_mask],
                           train_labels[new_node_val_mask])

  new_node_test_mask = np.array(
    [(a in new_node_set or b in new_node_set) for a, b in zip(test_sources, test_destinations)])
  new_node_test_data = Data(test_sources[new_node_test_mask],
                            test_destinations[new_node_test_mask],
                            test_timestamps[new_node_test_mask],
                            test_edge_idxs[new_node_test_mask],
                            test_labels[new_node_test_mask])

  # print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
  #                                                                     full_data.n_unique_nodes))
  # print("The training dataset has {} interactions, involving {} different nodes".format(
  #   train_data.n_interactions, train_data.n_unique_nodes))
  # print("The validation dataset has {} interactions, involving {} different nodes".format(
  #   val_data.n_interactions, val_data.n_unique_nodes))
  # print("The test dataset has {} interactions, involving {} different nodes".format(
  #   test_data.n_interactions, test_data.n_unique_nodes))
  # print("The new node validation dataset has {} interactions, involving {} different nodes".format(
  #   new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  # print("The new node test dataset has {} interactions, involving {} different nodes".format(
  #   new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  # print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
  #   len(new_test_node_set)))

  return full_node_features, full_edge_features, full_data, train_data, val_data, test_data_true, test_data_false, \
         new_node_val_data, new_node_test_data


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
