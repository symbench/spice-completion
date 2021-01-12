import numpy as np
import math

"""
This example reuses the Graph Attn Network from the paper:

Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""

from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data import DisjointLoader
from spektral.layers import GATConv

import datasets
import sys
import argparse
from model import model

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('--name', default='train_actions')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.005, type=float)
args = parser.parse_args()

# Configure tensorboard stuff
logdir = f'logs/{args.name}/' + datetime.now().strftime('%Y%m%d-%H%M%S')
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
# TODO: save experiment parameters

batch_size = args.batch_size
epochs = args.epochs

# Load data
dataset = datasets.omitted_with_actions(args.files)
#np.set_printoptions(threshold=100000)

# Train/valid/test split
idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
dataset_tr = dataset[idx_tr]
dataset_va = dataset[idx_va]
dataset_te = dataset[idx_te]

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(dataset_va, batch_size=batch_size)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size)

# Parameters
channels = 8            # Number of channel in each head of the first GAT layer
n_attn_heads = 8        # Number of attention heads in first GAT layer
F = dataset.n_node_features
dropout = 0.6           # Dropout rate for the features and adjacency matrix
dropout = 0.  # FIXME: remove
l2_reg = 5e-6           # L2 regularization rate
learning_rate = args.lr
epochs = args.epochs
es_patience = 100       # Patience for early stopping

# Model definition
loss_fn = CategoricalCrossentropy()
opt = Adam(lr=learning_rate)
#model.compile(optimizer=optimizer,
              #weighted_metrics=['acc'])
acc_fn = CategoricalAccuracy()
model.summary()

def forward(inputs, target):
    nodes, adj, edges = inputs
    output = model((nodes, adj), training=True)
    lens = [ len(graph_y) for graph_y in target ]

    output = tf.squeeze(output, axis=1)
    output = tf.RaggedTensor.from_row_lengths(output, lens)
    flat_targets = np.hstack(target)
    target_rt = tf.RaggedTensor.from_row_lengths(flat_targets, lens)
    mask = tf.math.not_equal(target_rt, -1)
    logits = tf.ragged.boolean_mask(output, mask)

    sums = tf.expand_dims(tf.reduce_sum(tf.math.exp(logits), 1), axis=1)
    action_probs = tf.divide(tf.math.exp(logits).to_tensor(), sums)
    target = tf.ragged.boolean_mask(target_rt, mask)
    target = tf.reshape(target.to_tensor(), action_probs.shape)

    return action_probs, target, mask

print('Fitting model')
current_batch = epoch = model_loss = model_acc = iteration = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
losses = []
accuracies = []
learning_layers_idx = None

def log_gradients(gradients):
    global learning_layers_idx 
    if learning_layers_idx is None:
        learning_layers_idx = [ i for (i, g) in enumerate(gradients) if np.linalg.norm(g) != 0 ]

    nonzero_grads = [ gradients[i] for i in learning_layers_idx ]
    grad_norm = sum((np.linalg.norm(g) for g in nonzero_grads)) / len(nonzero_grads)
    tf.summary.scalar('mean gradient norm', data=grad_norm, step=iteration)


# Train model
#@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        action_probs, target, _ = forward(inputs, target)

        loss = loss_fn(target, action_probs)
        loss += sum(model.losses)
        acc = acc_fn(target, action_probs)
        acc_fn.reset_states()
    gradients = tape.gradient(loss, model.trainable_variables)
    log_gradients(gradients)
    # TODO: clip gradients?
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return action_probs, target, loss, acc

def evaluate(loader, ops_list):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        (nodes, adj, edges), target = loader.__next__()
        pred = model((nodes, adj), training=False)
        outs = [o(target, pred) for o in ops_list]
        output.append(outs)
    return np.mean(output, 0)

for batch in loader_tr:
    target = batch[1]
    batch_size = target.shape[0]
    preds, targets, loss, acc = train_step(*batch)

    tf.summary.scalar('loss', data=loss, step=iteration)
    tf.summary.scalar('accuracy', data=acc, step=iteration)

    model_loss += loss
    model_acc += acc
    current_batch += 1
    iteration += 1
    losses.append(loss)
    accuracies.append(acc)
    if current_batch == loader_tr.steps_per_epoch:
        model_loss /= loader_tr.steps_per_epoch
        model_acc /= loader_tr.steps_per_epoch
        epoch += 1

        # Compute validation loss and accuracy
        print('Ep. {} - Loss: {:.2f} - Acc: {:.2f}'.format(epoch, model_loss, model_acc))
        #val_loss, val_acc = evaluate(loader_va, [loss_fn, acc_fn])
        #print('Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}'
              #.format(epoch, model_loss, model_acc, val_loss, val_acc))

        # Check if loss improved for early stopping
        #if val_loss < best_val_loss:
            #best_val_loss = val_loss
            #patience = es_patience
            #print('New best val_loss {:.3f}'.format(val_loss))
            #best_weights = model.get_weights()
        #else:
            #patience -= 1
            #if patience == 0:
                #print('Early stopping (best val_loss: {})'.format(best_val_loss))
                #break
        model_loss = 0
        model_acc = 0
        current_batch = 0

print('-------- Evaluating --------')
def select_prototype_types(prototype_types, actions):
    node_count = actions.shape[1]
    pred_idx = np.array([idx + i*node_count for (i, idx) in enumerate(np.argmax(actions, axis=1))])
    pred_types = np.take(prototype_types, pred_idx)
    return pred_types

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=1)
all_pred_types = []
all_actual_types = []
for batch in loader_tr:
    nodes, adj, edges = batch[0]
    actions, targets, mask = forward(*batch)
    node_types = np.argmax(nodes, axis=1)
    flat_mask = np.hstack(mask)
    prototype_types = tf.boolean_mask(node_types, flat_mask)

    pred_types = select_prototype_types(prototype_types, actions)
    actual_types = select_prototype_types(prototype_types, targets)

    all_pred_types.extend(pred_types)
    all_actual_types.extend(actual_types)

unique, counts = np.unique(all_actual_types, return_counts=True)
label_dist = dict(zip(unique, counts))
print('label distribution:')
for (key, value) in label_dist.items():
    print(f'{key}: {value}')

# confusion matrix
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
data = {'actual': all_actual_types, 'predicted': all_pred_types}
df = pd.DataFrame(data, columns=['actual', 'predicted'])
cm = pd.crosstab(df['actual'], df['predicted'], rownames=['Actual'], colnames=['Predicted'])
for idx in all_actual_types:
    if idx not in all_pred_types:
        cm[idx] = 0

totals = [ sum(row) for (_, row) in cm.iterrows() ]
cm['Totals'] = totals
sorted_cols = sorted([ c for c in cm.columns if type(c) is int ])
sorted_cols.append('Totals')
cm = cm.reindex(sorted_cols, axis=1)

sn.heatmap(cm, annot=True)
plt.savefig(f'{logdir}/confusion_matrix.png')
plt.show()

# Print summarization figures, stats
from matplotlib import pyplot as plt
plt.plot(accuracies)
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'])
plt.savefig(f'{logdir}/model_accuracy.png')

plt.clf()
plt.plot(losses)
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.savefig(f'{logdir}/model_loss.png')

# save the model
model.save(f'{logdir}/model')
print(f'saved model to {logdir}/model')
