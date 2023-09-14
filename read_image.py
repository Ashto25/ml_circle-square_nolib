import struct
import imageio
import numpy as np


def read(file):
    #Read in png file
    img = imageio.imread(file, mode='L')
    #Flatten the image
    img = img.flatten()
    #Normalize
    img = img/255
    #Map to 1 or 0
    img[img != 1] = 0

    return np.array(img)



def img_getsize(file):
     with open(file, 'rb') as d:
        data = b'' + d.read()
        #IHDR block of .png files containing image dimension information
        ihdr = data[12:36]
        x_size = struct.unpack('>i', ihdr[4:8])[0]
        y_size = struct.unpack('>i', ihdr[8:12])[0]
        print(f"Img is {x_size} by {y_size}")
     
        return x_size * y_size
     


def img_toconsole(flattened_data):
    i=0
    for v in flattened_data:
        if i%28==0:
            print("")
        elif v == 1:
            print("  ", end="")
        else:
            print("W", end="") 
        i+=1