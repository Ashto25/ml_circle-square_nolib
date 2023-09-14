import numpy as np
import load_data

#28 by 28 pixel images, black and white (0,1)
#Data is pre-processed by retrieve_data()
data = load_data.retrieve_data()

#Model architecture:
#    Input layer - single dimension layer with 32 nodes
#    Hidden layer 1 - fully connected layer with 64 nodes
#    Output layer - 2 node output layer mapping to either circle, or square


