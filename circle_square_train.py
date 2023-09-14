import numpy as np
import load_data
import read_image
import math
import random

#28 by 28 pixel images, black and white (0,1)
#Data is pre-processed by retrieve_data()
data, labels = load_data.retrieve_data()
input_datasize = read_image.img_getsize('shapes/circles/drawing(1).png')

#Train, eval split
train_splt = math.floor( 0.6 *len(data))

train_data = data[0:train_splt]
train_labels = labels[0:train_splt]

eval_data = data[train_splt:]
eval_labels = labels[train_splt:]

#Model architecture:
#    Input layer - single dimension layer with 784 nodes
#    Hidden layer 1 - fully connected layer with 200 nodes
#    Output layer - 2 node output layer mapping to either circle, or square


layer1_size = 100
bias1 = random.randint(0,1)
layer1_weights = []
for x in range(layer1_size):
    layer1_weights.append(np.random.rand(input_datasize) * 0.2 - 0.1)

output_size = 2
biasout = random.randint(0,1)
layerout_weights = []
for x in range(output_size):
    layerout_weights.append(np.random.rand(layer1_size) * 0.2 - 0.1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(logits):
    # Subtract the maximum logit value for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    softmax_probs = exp_logits / sum_exp_logits
    return softmax_probs

def predict_layer(input, actual):
    forward_pass1 = []
    for x in range(layer1_size):
        forward_pass1.append(sigmoid( np.dot(input, layer1_weights[x]) + bias1 ))

    forward_passout = []
    for x in range(output_size):
        forward_passout.append(sigmoid( np.dot(forward_pass1, layerout_weights[x]) + biasout ))

    output = softmax(forward_passout)
    print(f"softmax: {output}")

    if output[0] > output[1] and actual == 1:
        print("W")
    elif output[0] < output[1] and actual == 0:
        print("W")
    else:
        print("Correct prediction!")


predict_layer(train_data[0], train_labels[0])
predict_layer(train_data[1], train_labels[1])