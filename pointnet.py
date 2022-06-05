import os
from utils import *

if __name__ == "__main__":
    for i in os.listdir('MATLAB/Point Cloud Dataset'):
        print('Directory:', i)
        for j in os.listdir('MATLAB/Point Cloud Dataset/' + i):
            print('File:', j)
            file = open('MATLAB/Point Cloud Dataset/' + i + '/' + j)
            data = np.genfromtxt('MATLAB/Point Cloud Dataset/' + i + '/' + j, delimiter=',')
            print(data)

# Main Network
inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)
