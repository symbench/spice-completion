import numpy as np
import math
from spektral.layers import GraphAttention, GlobalAttentionPool

"""
This example reuses the Graph Attn Network from the paper:

Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from parse_netlist import load_omitted_netlists, is_valid_netlist
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
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
netlists = ( open(f, 'rb').read().decode('utf-8', 'ignore') for f in args.files )
netlists = ( text for text in netlists if is_valid_netlist(text) )
A, X, y = load_omitted_netlists(netlists)

train_A, train_X, train_y, indices = sample(A, X, y, 0.8)
val_A, val_X, val_y, indices = sample(A, X, y, 0.5, indices)
test_A, test_X, test_y, indices = sample(A, X, y, 1.0, indices)


# Parameters
channels = 8            # Number of channel in each head of the first GAT layer
n_attn_heads = 8        # Number of attention heads in first GAT layer
N = train_X.shape[0]          # Number of nodes in the graph
F = train_X.shape[-1]          # Original size of node features
n_classes = train_y.shape[-1]  # Number of classes
l2_reg = 5e-4           # L2 regularization rate
learning_rate = 5e-3    # Learning rate
epochs = args.epochs
es_patience = 100       # Patience for early stopping

# Preprocessing operations
A = A.astype('f4')
#X = X.toarray()

# Model definition
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))

gc1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in])
gc2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1, A_in])
pool = GlobalAttentionPool(128)(gc2)

output = Dense(n_classes, activation='softmax')(Flatten()(pool))

# Build model
model = Model(inputs=[X_in, A_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
unknowns = train_X.argmin(2)
print(train_X.shape)
print(train_A.shape)
print(train_y.shape)
print(f'about to train on {train_X.shape[0]} points ({X.shape[0]} total)')
validation_data = ([val_X, val_A], val_y)
history = model.fit([train_X, train_A],
          train_y,
          #sample_weight=unknowns,
          epochs=epochs,
          #batch_size=N,
          validation_data=validation_data,
          shuffle=True,
          )

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

import scikitplot as skplt
def plot_confusion_matrix(A, X, y):
    plt.clf()
    y_pred = model.predict([X, A])
    #pred_unk = X.argmin(2)*y_pred.argmax(2)
    #act_unk = X.argmin(2)*y.argmax(2)
    #y_pred = pred_unk[pred_unk.nonzero()]
    #y_act = act_unk[act_unk.nonzero()]
    y_pred = y_pred.argmax(1)
    y_act = y.argmax(1)
    print('  predicted:\t', y_pred)
    print('  actual:\t', y_act)
    skplt.metrics.plot_confusion_matrix(y_act, y_pred)

print('Generating confusion matrices...')
print('train:')
plot_confusion_matrix(train_A, train_X, train_y)
plt.savefig(f'train_confusion_matrix_{args.name}.png')

print('val:')
plot_confusion_matrix(val_A, val_X, val_y)
plt.savefig(f'val_confusion_matrix_{args.name}.png')

print('test:')
plot_confusion_matrix(test_A, test_X, test_y)
plt.savefig(f'test_confusion_matrix_{args.name}.png')

print(f'trained on {train_X.shape[0]} points ({X.shape[0]} total)')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([test_X, test_A], test_y)

print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))

