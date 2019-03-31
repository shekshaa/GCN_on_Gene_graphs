import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from load_dataset import load_classes2
import pandas as pd
import numpy as np


base_path = './random_data/'
labels, one_hot_labels, num_graphs, num_classes = load_classes2(base_path)
embeddings_df = pd.read_csv('./embedding/graph_embedding20.csv')
cols = embeddings_df.columns[1:]
embeddings = embeddings_df[cols].values

num_samples = num_graphs
idx = np.arange(num_samples)
np.random.shuffle(idx)

sampled_embeddings = embeddings[idx]
sampled_labels = [labels[i] for i in idx]

reduced_embedding = TSNE(n_components=2).fit_transform(sampled_embeddings)
color_names = ['blue', 'green', 'red', 'yellow', 'cyan', 'black', 'orange', 'magenta', 'silver', 'lime']
colors = [color_names[label] for label in sampled_labels]
plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1],
            marker='.',
            c=colors)
plt.show()
