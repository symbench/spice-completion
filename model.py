import tensorflow.keras as keras
from tensorflow.keras.models import Model
from spektral.layers import *
from tensorflow.keras.layers import *

def custom_elu_alpha_12(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_1(x):
    return keras.activations.elu(x, alpha=1)

nodes_output = Input(shape=(29, ), batch_size=None, dtype=None, sparse=False, tensor=None, ragged=False)
nodes_normalized = BatchNormalization()(nodes_output)
node_embeddings = Dense(100)(nodes_normalized)
adjacency_output = Input(shape=(None, ), sparse=True)
graphattention = GATConv(channels=100, attn_heads=1, concat_heads=True, dropout_rate=0., return_attn_coef=False, use_bias=True, kernel_initializer=keras.initializers.GlorotUniform(seed=None), bias_initializer=keras.initializers.Zeros(), attn_kernel_initializer=keras.initializers.GlorotUniform(seed=None))

graphattention_output = graphattention(inputs=[node_embeddings, adjacency_output])

sharedweightlayer_output = graphattention(inputs=[graphattention_output, adjacency_output])
sharedweightlayer_output = graphattention(inputs=[sharedweightlayer_output, adjacency_output])
sharedweightlayer_output = graphattention(inputs=[sharedweightlayer_output, adjacency_output])

graphattention2 = GATConv(channels=1, attn_heads=1, concat_heads=True, dropout_rate=0., return_attn_coef=False, activation=custom_elu_alpha_1, use_bias=True, kernel_initializer=keras.initializers.GlorotUniform(seed=None), bias_initializer=keras.initializers.Zeros(), attn_kernel_initializer=keras.initializers.GlorotUniform(seed=None))
graphattention_output2 = graphattention2(inputs=[sharedweightlayer_output, adjacency_output])
flatten = Flatten(data_format=None)
flatten_output = flatten(inputs=graphattention_output2)
activation = Activation(activation=custom_elu_alpha_12)
activation_output = activation(inputs=flatten_output)

custom_objects = {}
custom_objects['custom_elu_alpha_1'] = custom_elu_alpha_1
custom_objects['custom_elu_alpha_12'] = custom_elu_alpha_12


model = Model(inputs=[nodes_output,adjacency_output], outputs=[activation_output])
result = model
model.custom_objects = custom_objects
