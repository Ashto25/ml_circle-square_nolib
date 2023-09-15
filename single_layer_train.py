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

learning_rate = 0.01
#Model architecture:
#    Input layer - single dimension layer with 784 nodes
#    Hidden layer 1 - fully connected layer with 200 nodes
#    Output layer - 2 node output layer mapping to either circle, or square



output_size = 1
biasout = []
layerout_weights = []
for x in range(output_size):
    layerout_weights.append(np.random.rand(input_datasize) * 0.2 - 0.1)
    biasout.append(random.randint(0,1))


def binary_cross_entropy_loss(predictions, targets):
    epsilon = 1e-15  # A small constant to avoid taking the log of zero
    y_pred = np.clip(predictions, epsilon, 1 - epsilon)  # Clip values to avoid extreme cases
    loss = - (targets * np.log(y_pred) + (1 - targets) * np.log(1 - y_pred))
    #loss = -[y * log(p) + (1 - y) * log(1 - p)]
    return loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(logits):
    # Subtract the maximum logit value for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    softmax_probs = exp_logits / sum_exp_logits
    return softmax_probs

def logit(p):
    return np.log(p / (1 - p))


def predict_layer(input, actual, train):

    forward_passout = []
    for x in range(output_size):
        forward_passout.append(sigmoid( np.dot(input, layerout_weights[x]) + biasout[x] ))


    output = forward_passout[0]

    loss_output = (binary_cross_entropy_loss(output, actual))
    #print(f"Loss: {loss_output}, Actual: {actual}, Prediction: {output}")

    if train:
        #grad_layerout_weights = (np.outer(forward_pass1, (loss_output-output)))
        for x in range(output_size):
            denom = (output * (1 - output))
            if denom == 0:
                denom = 0.0001
            dLda = (output - actual) / denom
            dadz = output * (1 - output) 
            dLdz = dLda * dadz
            #print(dLdz)
        
            for w in range(input_datasize):
                dzdw=input[w]
                dLdw = (dLdz) * (dzdw)
                #print(dLdw)
            #update = (learning_rate * grad_layerout_weights[x])
                layerout_weights[x][w] -= (learning_rate * dLdw)
            #biasout[x] -= learning_rate * (loss_output-output)


    if (output >= 0.5 and actual == 1):
        return 1
    elif (output < 0.5 and actual == 0):
        return 1
    return 0


epochs = 50
runs_per = len(eval_data)-1#60

def eval():
    sum=0
    for x in range(len(eval_data)):
        result = predict_layer(eval_data[x], eval_labels[x], False)
        sum+=result

    print(f"  Final EVAL: {sum/(len(eval_data))}")

for e in range(epochs):
    sum=0
    for i in range(runs_per):
        x = i#random.randint(0,len(train_data)-1)
        result = predict_layer(train_data[x], train_labels[x], True)
        sum+=result
    print(f"Final out is: {sum/(runs_per)}", end='')
    eval()


