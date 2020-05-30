import numpy as np
import time
import os
from scipy import misc
from PIL import Image


class point_cloud_generator():
    def __init__(self, depth_file):
        self.depth_file = depth_file
        self.depth = Image.open(depth_file).convert('I')
        self.width = self.depth.size[1]
        self.height = self.depth.size[0]

    def calculate(self):
        t1 = time.time()
        depth = np.asarray(self.depth)
        Z = np.zeros((self.width, self.height))
        for i in range(self.width):
            for j in range(self.height):
                Z[i, j] = 255 - depth[i, j]
        self.Z = Z
        df = np.zeros((self.width, self.height))
        for i in range(self.width):
            df[i] = Z[i, :]
        self.df = df
        t2 = time.time()
        print('calcualte Done.', t2 - t1)
        return df


def main():
    save_path = ""
    depth_path = ""
    depths = [depth_path + f for f in os.listdir(depth_path) if f.endswith('.png')]
    depths = sorted(depths)
    for depth in depths:
        name = depth.split('/')[-1][:-4]
        a = point_cloud_generator(depth)
        df = a.calculate()
        misc.imsave(save_path + name + '.png', df)
        print("save success", name)


if __name__ == "__main__":
    main()
