import csv
import os
import trimesh

if __name__ == "__main__":
    for i in os.listdir('MATLAB/Point Cloud Dataset'):
        print('Directory:', i)
        for j in os.listdir('MATLAB/Point Cloud Dataset/' + i):
            print('File:', j)
            file = open('MATLAB/Point Cloud Dataset/' + i + '/' + j)
            tri = trimesh.load('MATLAB/Point Cloud Dataset/' + i + '/' + j)
            print(tri)

            # Convert csv into python arrays
            # csv_reader = csv.reader(file)
            # rows = []
            # for row in csv_reader:
            #     rows.append(row)
            #
            # point_set = []
            # for index, data_point in enumerate(rows[0]):
            #     point = [data_point, rows[1][index], rows[2][index], rows[3][index]]
            #     point_set.append(point)
            #     s = trimesh.Trimesh(point)
            #     print(type(s))
            #     print(s)
            #     break
            # print(point_set)