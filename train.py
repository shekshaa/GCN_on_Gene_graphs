from load_dataset import *
from utils import *
from model import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


flags.DEFINE_string('model', 'inception', 'Model string.')  # gcn, gcn_cheby, inception
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 36, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 18, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 9, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_bool('featureless', False, 'featureless')


base_path = './data4/'
adj, num_nodes = load_adj(base_path)
labels, one_hot_labels, num_graphs, num_classes = load_classes2(base_path)
class_dist = [labels.tolist().count(i) / num_graphs for i in range(num_classes)]

print(class_dist)
features = load_features(base_path, is_binary=False)

# num_train = num_graphs
train_proportion = 0.7
num_train = int(num_graphs * train_proportion)
idx = np.arange(num_graphs)
np.random.shuffle(idx)

# indices of train and test
train_idx = idx[:num_train]
test_idx = idx[num_train:]

# collecting train samples
train_labels = labels[train_idx]
train_one_hot_labels = one_hot_labels[train_idx]
train_features = features[train_idx]

# collecting test samples
test_labels = labels[test_idx]
test_one_hot_labels = one_hot_labels[test_idx]
test_features = features[test_idx]

train_sparse_features = []
test_sparse_features = []
sparse_features = []

train_graph_weights = [1 / class_dist[train_labels[i]] for i in range(num_train)]

for i in range(num_graphs):
    sparse_features.append(sparse_to_tuple(sp.coo_matrix(np.expand_dims(np.transpose(features[i, :]), 1))))

for i in range(num_train):
    train_sparse_features.append(sparse_to_tuple(sp.coo_matrix(np.expand_dims(np.transpose(train_features[i, :]), 1))))

for i in range(num_graphs - num_train):
    test_sparse_features.append(sparse_to_tuple(sp.coo_matrix(np.expand_dims(np.transpose(test_features[i, :]), 1))))

if FLAGS.model == 'gcn_cheby':
    locality1 = 8
    locality2 = 7
    locality3 = 6
    locality = [locality1, locality2, locality3]  # locality sizes of different blocks
    num_supports = np.max(locality) + 1
    support = chebyshev_polynomials(adj, num_supports - 1)
elif FLAGS.model == 'inception':
    locality_sizes = [7, 5, 3]
    num_supports = np.max(locality_sizes) + 1
    support = chebyshev_polynomials(adj, num_supports - 1)
elif FLAGS.model == 'gcn':
    num_supports = 1
    support = [preprocess_adj(adj)]
else:
    raise NotImplementedError


placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32),
        'labels': tf.placeholder(tf.float32, shape=(one_hot_labels.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'weight': tf.placeholder(tf.float32),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

# model definition
if FLAGS.model == 'gcn_cheby':
    model = CheybyGCN(placeholders, input_dim=1, num_class=num_classes, locality=locality)
elif FLAGS.model == 'inception':
    model = InceptionGCN(placeholders, input_dim=1, num_class=num_classes,
                         locality_sizes=locality_sizes, is_pool=True)
else:
    model = SimpleGCN(placeholders, input_dim=1, num_class=num_classes)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_result = []
    for epoch in range(FLAGS.epochs):
        # print('Starting epoch {}'.format(epoch + 1))
        cnt = 0
        sum_loss = 0
        train_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
        test_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
        for i in range(num_train):
            train_feed_dict = construct_feed_dict(train_sparse_features[i], support, train_one_hot_labels[i],
                                                  train_graph_weights[i], placeholders)
            train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            _, loss, acc, out = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs],
                                         feed_dict=train_feed_dict)
            train_acc_classes[train_labels[i], np.argmax(out, 1)[0]] += 1
            # print('Graph {}: '.format(i + 1), 'Loss={}, '.format(loss), 'Acc={}'.format(acc))
            # train_acc_classes[train_labels[i], prediction] += 1
            cnt += acc
            sum_loss += loss
        print('Epoch {}:'.format(epoch + 1), 'acc={:.4f}, loss={:.4f}'.format(cnt / float(num_train),
              sum_loss / float(num_train)))

        cnt = 0
        for i in range(num_graphs - num_train):
            test_feed_dict = construct_feed_dict(test_sparse_features[i], support, test_one_hot_labels[i], 1,
                                                 placeholders)
            test_feed_dict.update({placeholders['dropout']: 0.})

            acc, out = sess.run([model.accuracy, model.outputs], feed_dict=test_feed_dict)
            test_acc_classes[test_labels[i], np.argmax(out, 1)[0]] += 1
            # test_acc_classes[test_labels[i], prediction] += 1
            cnt += acc
        test_acc = cnt / float(num_graphs - num_train)
        test_result.append(test_acc)
        print('Test accuracy: {:.4f}'.format(test_acc))
        print('train confusion matrix: \n', train_acc_classes)
        print('test confusion matrix: \n', test_acc_classes)

    print("Optimization finished!")

    plt.plot(test_result)
    plt.show()
    print('Storing graph embedding')
    embedding_level = 4
    with open('./embedding/graph_embedding21.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        header = ['id']
        for i in range(FLAGS.hidden3):
            header.append('emb_{}'.format(i))
        writer.writerow(header)
        embeddings = []
        for i in range(num_graphs):
            feed_dict = construct_feed_dict(sparse_features[i], support, one_hot_labels[i], 1, placeholders)
            feed_dict.update({placeholders['dropout']: 0.})
            embedding = sess.run(model.activations[embedding_level], feed_dict=feed_dict)
            row = [i + 1]
            for item in embedding.tolist()[0]:
                row.append(item)
            writer.writerow(row)
            embeddings.append(embedding.tolist()[0])

    # print('Plotting t-SNE')
    # embeddings = np.asarray(embeddings)
    # reduced_embedding = TSNE(n_components=2).fit_transform(embeddings)
    # color_names = ['b', 'g', 'r', 'y']
    # colors = [color_names[label] for label in labels]
    # num_samples = 5000
    # plt.scatter(reduced_embedding[:num_samples, 0], reduced_embedding[:num_samples, 1],
    #             marker='.',
    #             c=colors[:num_samples])
    # plt.show()
