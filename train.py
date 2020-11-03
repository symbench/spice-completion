import numpy as np
import math
from spektral.layers import GraphAttention

"""
This example implements the experiments on citation networks from the paper:

Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphAttention

from parse_netlist import load_netlist
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file')  # FIXME: add varargs , nargs='+')
parser.add_argument('--name', default='train')
parser.add_argument('--epochs', default=100, type=int)
args = parser.parse_args()

def sample(A, X, y, amt, indices=None):
    if indices is None:
        indices = np.arange(X.shape[0])
    split_size = math.floor(indices.size*amt)
    split_idx = np.random.choice(indices, split_size, replace=False)
    split_X = X[split_idx]
    split_y = y[split_idx]
    split_A = A[split_idx]
    indices = np.setdiff1d(indices, split_idx)
    return split_A, split_X, split_y, indices

# Load data
with open(args.file, 'rb') as f:
    A, X, y = load_netlist(f.read().decode('utf-8', 'ignore'))

train_A, train_X, train_y, indices = sample(A, X, y, 0.8)
val_A, val_X, val_y, indices = sample(A, X, y, 0.5, indices)
test_A, test_X, test_y, indices = sample(A, X, y, 1.0, indices)

# Parameters
channels = 8            # Number of channel in each head of the first GAT layer
n_attn_heads = 8        # Number of attention heads in first GAT layer
N = train_X.shape[0]          # Number of nodes in the graph
F = train_X.shape[-1]          # Original size of node features
n_classes = train_y.shape[-1]  # Number of classes
dropout = 0.6           # Dropout rate for the features and adjacency matrix
dropout = 0.  # FIXME: remove
l2_reg = 5e-6           # L2 regularization rate
learning_rate = 5e-3    # Learning rate
epochs = args.epochs
es_patience = 100       # Patience for early stopping

# Preprocessing operations
A = A.astype('f4')
#X = X.toarray()

# Model definition
X_in = Input(shape=(F, ))
A_in = Input(shape=(N, ))

dropout_1 = Dropout(dropout)(X_in)
graph_attention_1 = GraphAttention(channels,
                                   attn_heads=n_attn_heads,
                                   concat_heads=True,
                                   dropout_rate=dropout,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg),
                                   name='firstGAT',
                                   )([dropout_1, A_in])
dropout_2 = Dropout(dropout)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes,
                                   attn_heads=1,
                                   concat_heads=False,
                                   dropout_rate=dropout,
                                   activation='softmax',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg),
                                   name='secondGAT',
                                   )([dropout_2, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=graph_attention_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
validation_data = ([val_X, val_A], val_y)
history = model.fit([train_X, train_A],
          train_y,
          #sample_weight=train_mask,
          epochs=epochs,
          #batch_size=N,
          validation_split=0.1,
          #validation_data=validation_data,
          shuffle=True,
          )
          #callbacks=[
              #EarlyStopping(patience=es_patience, restore_best_weights=True)
          #])

from matplotlib import pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'])
plt.savefig(f'model_accuracy_{args.name}.png')

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.savefig(f'model_loss_{args.name}.png')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([test_X, test_A], test_y)
                              #test_y)
                              #sample_weight=test_mask,

print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))

y_pred = model.predict([train_X, train_A])
print('y_pred shape', y_pred.shape)
print('predicted shape', y_pred.argmax(2).shape)
print('actual:\n', train_y.argmax(2))
print('predicted:\n', y_pred.argmax(2))
print('unknowns:\n', train_X.argmin(2))
pred_unk = train_X.argmin(2)*y_pred.argmax(2)
act_unk = train_X.argmin(2)*train_y.argmax(2)
print('predicted unknowns:\n', pred_unk)
print('actual unknowns:\n', act_unk)

y_pred = pred_unk[pred_unk.nonzero()]
y_act = act_unk[act_unk.nonzero()]
print('predicted unknowns:\n', y_pred)
print('actual unknowns:\n', y_act)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_act, y_pred)
plt.savefig(f'confusion_matrix_{args.name}.png')
