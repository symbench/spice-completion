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
args = parser.parse_args()

# Load data
with open(args.file, 'rb') as f:
    A, X, y, train_mask, val_mask, test_mask = load_netlist(f.read().decode('utf-8', 'ignore'))

# Parameters
channels = 8            # Number of channel in each head of the first GAT layer
n_attn_heads = 8        # Number of attention heads in first GAT layer
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original size of node features
n_classes = y.shape[1]  # Number of classes
dropout = 0.6           # Dropout rate for the features and adjacency matrix
dropout = 0.  # FIXME: remove
l2_reg = 5e-6           # L2 regularization rate
learning_rate = 5e-3    # Learning rate
epochs = 20000          # Number of training epochs
epochs = 150  # FIXME: remove
es_patience = 100       # Patience for early stopping

# Preprocessing operations
A = A.astype('f4')
#X = X.toarray()

# Model definition
X_in = Input(shape=(F, ))
A_in = Input(shape=(N, ), sparse=True)

dropout_1 = Dropout(dropout)(X_in)
graph_attention_1 = GraphAttention(channels,
                                   attn_heads=n_attn_heads,
                                   concat_heads=True,
                                   dropout_rate=dropout,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg)
                                   )([dropout_1, A_in])
dropout_2 = Dropout(dropout)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes,
                                   attn_heads=1,
                                   concat_heads=False,
                                   dropout_rate=dropout,
                                   activation='softmax',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg)
                                   )([dropout_2, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=graph_attention_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
validation_data = ([X, A], y, val_mask)
history = model.fit([X, A],
          y,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
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
eval_results = model.evaluate([X, A],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)

print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))

y_pred = model.predict([X, A])
print('actual:\n', y.argmax(1))
print('predicted:\n', y_pred.argmax(1))
print('train mask:\n', train_mask.astype(int))
print('test mask:\n', test_mask.astype(int))
# TODO: Generate confusion matrix?
