import numpy as np
import read_image
def retrieve_data():
    #Data storage
    data = np.empty(0)
    #Load 100 squares and circles
    for i in range(100):
        #Load in the images
        square = read_image.read("/shapes/squares/drawing("+str(i)+").png")
        circle = read_image.read("/shapes/circles/drawing("+str(i)+").png")
        #Flatten the pixel data


        
        #Add to data storage: 0 is circle, 1 is square
        np.append(data, (circle, 0))
        np.append(data, (square, 1))
