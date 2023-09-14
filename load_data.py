import numpy as np
import read_image
def retrieve_data():
    #Data storage
    data = []
    labels = []
    #Load 100 squares and circles
    for i in range(1,100):
        #Load in the images
        square = read_image.read("shapes/squares/drawing("+str(i)+").png")
        circle = read_image.read("shapes/circles/drawing("+str(i)+").png")
        #Add to data storage: 0 is circle, 1 is square
        data.append(square)
        labels.append(1)
        data.append(circle)
        labels.append(0)
    return data, labels
