
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

# Load data
dataset = 'cora'
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

print('A:')
print(A.shape)
print('X:')
print(type(X))
print(X.shape)
print('y:')
print(y.shape)
print('train_mask:')
print(train_mask.shape)
print('True values in train mask:', len([ i for i in list(train_mask) if i ]))
print('True values in val mask:', len([ i for i in list(val_mask) if i ]))
print('True values in test mask:', len([ i for i in list(test_mask) if i ]))
