from pathlib import Path
from typing import List, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = TypeVar('S')

class WeightVectorNeighborhood:
    """ 权重向量邻域 (MOEA/D) """

    def __init__(self, 
            number_of_weight_vectors: int,  # 权重向量个数（即种群大小）
            neighborhood_size: int,  # 邻域大小（每个解的邻居个数）
            weight_vector_size: int = 2,  # 权重向量维度（2维/3维）
            weights_path: str = None  # 权重向量文件路径
        ):
        self.number_of_weight_vectors = number_of_weight_vectors
        self.neighborhood_size = neighborhood_size
        self.weight_vector_size = weight_vector_size
        self.weights_path = weights_path

        self.weight_vectors = np.zeros((number_of_weight_vectors, weight_vector_size))  # 权重向量矩阵
        self.neighborhood = np.zeros((number_of_weight_vectors, neighborhood_size), dtype=int)  # 邻居矩阵
        self.__initialize_uniform_weight()  # 初始化权重向量矩阵
        self.__initialize_neighborhood()  # 初始化邻居矩阵

    def __initialize_uniform_weight(self) -> None:
        """ 初始化权重向量矩阵: 从文件中读取 """
        file_name = 'W{}D_{}.dat'.format(self.weight_vector_size, self.number_of_weight_vectors)
        file_path = self.weights_path + '/' + file_name

        if Path(file_path).is_file():
            with open(file_path) as file:
                for index, line in enumerate(file):
                    vector = [float(x) for x in line.split()]
                    self.weight_vectors[index][:] = vector
        else:
            raise FileNotFoundError(
                'Failed to initialize weights: {} not found, please generate these weights first.'.format(file_path))

    def __initialize_neighborhood(self) -> None:
        """ 初始化邻居矩阵 """
        distance = np.zeros((len(self.weight_vectors), len(self.weight_vectors)))  # 初始化距离矩阵

        for i in range(len(self.weight_vectors)):
            for j in range(len(self.weight_vectors)):
                distance[i][j] = np.linalg.norm(self.weight_vectors[i] - self.weight_vectors[j])  # 计算权重间的欧氏距离

            indexes = np.argsort(distance[i, :])  # 根据欧氏距离从小到大排序
            self.neighborhood[i, :] = indexes[0:self.neighborhood_size]  # 找到各自的邻域，即欧氏距离最小的一系列权重向量的index

    def get_neighbors(self, index: int, solution_list: List[S]) -> List[S]:
        """ 获取指定index解的邻域解集（一个index对应一个解，也对应着一个权重向量） """
        neighbors_indexes = self.neighborhood[index]  # 获取指定解的邻居集合（均为index）

        if any(i > len(solution_list) for i in neighbors_indexes):
            raise IndexError('Neighbor index out of range')

        return [solution_list[i] for i in neighbors_indexes]  # 获取index在种群中对应的solution，返回solution组成的邻居解集

    def get_neighborhood(self):
        """ 获取邻居矩阵 """
        return self.neighborhood


class weights_generator(object):
    """
    权重向量产生器

    参考：https://blog.csdn.net/jiang425776024/article/details/84528415
    根据任意的种群大小产生均匀分布的权重向量 --- 无法实现
    当前只能通过指定在每个目标函数轴上的采样个数，来产生均匀分布的权重向量，产生的个数是不受控制的！
    """
    def perm(self, sequence):
        l = sequence
        if len(l) <= 1:
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def generate_weights(self, H, m):
        """
        :param H: # 每个目标方向上的采样个数（不包括原点）
        :param m: 权重向量的维数（即：目标维数）
        :return weight_vectors: 均匀分布的权重向量（np数组）
        """
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        weight_vectors = []
        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in weight_vectors:
                weight_vectors.append(weight)

        return np.array(weight_vectors)

    @staticmethod
    def save(weights):
        file_path = r'..\resources\MOEAD_weights\W{}D_{}.dat' \
            .format(weights.shape[1], weights.shape[0])
        # 注意虽然是to_csv方法，但是由于指定了文件后缀为dat，保存出来仍为dat格式
        pd.DataFrame(weights).to_csv(file_path, sep=' ', header=None, index=False)

    @staticmethod
    def visulize(weights):
        if weights.shape[1] == 2:
            plt.figure()
            plt.scatter(weights[:, 0], weights[:, 1])
            plt.show()
        elif weights.shape[1] == 3:
            plt.figure()
            ax = plt.subplot(projection='3d')
            ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2])
            plt.show()
        else:
            raise KeyError('Only 2D or 3D weights can be visualized, but weights is {}D.'.format(weights.shape[1]))


if __name__ == '__main__':
    # 生成权重向量，并保存为文件、可视化展示
    generator = weights_generator()
    for i in range(80):
        w = generator.generate_weights(i + 1, 3)
        generator.save(w)
        # generator.visulize(w)

