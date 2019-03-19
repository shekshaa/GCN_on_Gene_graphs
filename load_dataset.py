import csv
import numpy as np
import os
import pandas as pd


def load_adj(path):
    full_path = os.path.join(path, 'adj.csv')
    num_nodes = -1
    adj = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            adj.append([float(row[i]) for i in range(1, len(row))])

    adj = np.asarray(adj)
    return adj, num_nodes


def load_classes(path):
    full_path = os.path.join(path, 'classes.csv')
    labels = []
    class_names = []
    num_graphs = -1
    num_classes = 0
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_graphs += 1
            if num_graphs == 0:
                continue
            if row[2] in class_names:
                labels.append(class_names.index(row[2]))
            else:
                class_names.append(row[2])
                labels.append(num_classes)
                num_classes += 1

    labels = np.asarray(labels)
    one_hot_labels = np.zeros((num_graphs, num_classes))
    one_hot_labels[np.arange(num_graphs), labels] = 1

    return labels, one_hot_labels, num_graphs, num_classes


def load_classes2(path):
    full_path = os.path.join(path, 'classes.csv')
    classes = pd.read_csv(full_path)
    classes.dropna(axis=0, inplace=True)
    labels = classes['id'].values.astype(int)
    num_classes = np.max(labels)
    num_graphs = labels.shape[0]
    labels -= np.ones(shape=(num_graphs,), dtype=int)
    one_hot_labels = np.zeros((num_graphs, num_classes))
    one_hot_labels[np.arange(num_graphs), labels] = 1
    return labels, one_hot_labels, num_graphs, num_classes


def load_features(path, is_binary=False):
    full_path = os.path.join(path, 'features.csv')
    num_nodes = -1
    features = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            if is_binary:
                features.append([1 if float(row[i]) > 0 else 0 for i in range(1, len(row))])
            else:
                features.append([float(row[i]) for i in range(1, len(row))])
    features = np.asarray(features)
    features = features.T
    return features
