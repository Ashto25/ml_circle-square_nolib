import numpy as np
import load_data
import read_image
import math
import random
import matplotlib.pyplot as plt

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
#    Output layer - 1 node output layer mapping to either circle, or square

output_size = 1
biasout = []
layerout_weights = []

for x in range(output_size):
    layerout_weights.append(np.random.rand(input_datasize) * 0.2 - 0.1)
    biasout.append(random.randint(0,1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_layer(input, actual, train):

    forward_passout = []
    for x in range(output_size):
        forward_passout.append(sigmoid( np.dot(input, layerout_weights[x]) + biasout[x] ))


    output = forward_passout[0]


    #loss_output = (binary_cross_entropy_loss(output, actual))
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
    evals = []
    sum=0
    for x in range(len(eval_data)):
        result = predict_layer(eval_data[x], eval_labels[x], False)
        sum+=result
        evals.append((x, result))
    print(f"  Final EVAL: {sum/(len(eval_data))}")

    offset=5
    fig, sp = plt.subplots(1, 6)
    for x in range(6):
        sp[x].imshow( np.reshape( eval_data[evals[x+offset][0]], (28,28) ), vmin=0, vmax=1 )
        if evals[x+offset][1]:
            sp[x].set_title("Correct")
        else:
            sp[x].set_title("Wrong")
        sp[x].set_axis_off()
    #plt.imshow(np.reshape(layerout_weights, (28,28)), vmin=0, vmax=1)
    plt.show()



for e in range(epochs):
    sum=0
    for i in range(runs_per):
        x = i#random.randint(0,len(train_data)-1)
        result = predict_layer(train_data[x], train_labels[x], True)
        sum+=result
    #print(f"Final out is: {sum/(runs_per)}", end='')
    #eval()

eval()

plt.imshow(np.reshape(layerout_weights, (28,28)), vmin=0, vmax=1)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()


