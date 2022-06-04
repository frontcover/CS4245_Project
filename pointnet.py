import os
import numpy as np

if __name__ == "__main__":
    for i in os.listdir('MATLAB/Point Cloud Dataset'):
        print('Directory:', i)
        for j in os.listdir('MATLAB/Point Cloud Dataset/' + i):
            print('File:', j)
            file = open('MATLAB/Point Cloud Dataset/' + i + '/' + j)
            data = np.genfromtxt('MATLAB/Point Cloud Dataset/' + i + '/' + j, delimiter=',')
            print(data)