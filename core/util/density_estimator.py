from abc import ABC, abstractmethod
from functools import cmp_to_key
from typing import List, TypeVar

import numpy
from scipy.spatial.distance import euclidean

from core.util.comparator import Comparator, SolutionAttributeComparator

S = TypeVar('S')

class DensityEstimator(List[S], ABC):
    """This is the interface of any density estimator algorithm.
    """

    @abstractmethod
    def compute_density_estimator(self, solutions: List[S]) -> float:
        pass

    @abstractmethod
    def sort(self, solutions: List[S]) -> List[S]:
        pass

    @classmethod
    def get_comparator(cls) -> Comparator:
        pass


class CrowdingDistance(DensityEstimator[List[S]]):
    """This class implements a DensityEstimator based on the crowding distance of algorithm NSGA-II.
    """

    def compute_density_estimator(self, front: List[S]):
        """This function performs the computation of the crowding density estimation over the solution list.

        .. note::
           This method assign the distance in the inner elements of the solution list.

        :param front: The list of solutions.
        """
        size = len(front)

        if size == 0:
            return
        elif size == 1:
            front[0].attributes['crowding_distance'] = float("inf")  # 只有一个个体，拥挤距离为无限大inf
            return
        elif size == 2:
            front[0].attributes['crowding_distance'] = float("inf")  # 只有两个个体，拥挤距离均为无限大inf
            front[1].attributes['crowding_distance'] = float("inf")  # 只有两个个体，拥挤距离均为无限大inf
            return

        # 大于等于3个个体时，开始正常计算拥挤距离
        for i in range(len(front)):
            front[i].attributes['crowding_distance'] = 0.0  # 首先初始化每个个体的拥挤距离为0

        number_of_objectives = front[0].number_of_objectives

        for i in range(number_of_objectives):
            # Sort the population by Obj n
            front = sorted(front, key=lambda x: x.objectives[i])  # 将front按照第i维目标函数从小到大排序
            objective_minn = front[0].objectives[i]  # 第i维目标函数的最小值
            objective_maxn = front[len(front) - 1].objectives[i]  # 第i维目标函数的最大值

            # Set de crowding distance
            front[0].attributes['crowding_distance'] = float('inf')  # 第i维目标函数最小的个体解，其拥挤距离设置为inf
            front[size - 1].attributes['crowding_distance'] = float('inf')  # 第i维目标函数最大的个体解，其拥挤距离设置为inf

            for j in range(1, size - 1):
                distance = front[j + 1].objectives[i] - front[j - 1].objectives[i]

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass;  # LOGGER.warning('Minimum and maximum are the same!')
                else:
                    distance = distance / (objective_maxn - objective_minn)

                distance += front[j].attributes['crowding_distance']
                front[j].attributes['crowding_distance'] = distance

    def sort(self, solutions: List[S]) -> List[S]:
        solutions.sort(key=cmp_to_key(self.get_comparator().compare))  # 排序依据：按照拥挤距离排序，越大排序越靠前

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("crowding_distance", lowest_is_best=False)  # 拥挤距离越大越好，因此设置为False


class KNearestNeighborDensityEstimator(DensityEstimator[List[S]]):
    """This class implements a density estimator based on the distance to the k-th nearest solution.
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k
        self.distance_matrix = []

    def compute_density_estimator(self, solutions: List[S]):
        solutions_size = len(solutions)
        if solutions_size <= self.k:
            return

        points = []
        for i in range(solutions_size):
            points.append(solutions[i].objectives)

        # Compute distance matrix
        self.distance_matrix = numpy.zeros(shape=(solutions_size, solutions_size))
        for i in range(solutions_size):
            for j in range(solutions_size):
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = euclidean(solutions[i].objectives,
                                                                                    solutions[j].objectives)
        # Gets the k-nearest distance of all the solutions
        for i in range(solutions_size):
            distances = []
            for j in range(solutions_size):
                distances.append(self.distance_matrix[i, j])
            distances.sort()
            solutions[i].attributes['knn_density'] = distances[self.k]

    def sort(self, solutions: List[S]) -> List[S]:
        def compare(solution1, solution2):
            distances1 = solution1.attributes["distances_"]
            distances2 = solution2.attributes["distances_"]

            tmp_k = self.k
            if distances1[tmp_k] > distances2[tmp_k]:
                return -1
            elif distances1[tmp_k] < distances2[tmp_k]:
                return 1
            else:
                while tmp_k < (len(distances1) - 1):
                    tmp_k += 1
                    if distances1[tmp_k] > distances2[tmp_k]:
                        return -1
                    elif distances1[tmp_k] < distances2[tmp_k]:
                        return 1
            return 0

        for i in range(len(solutions)):
            distances = []
            for j in range(len(solutions)):
                distances.append(self.distance_matrix[i, j])
            distances.sort()
            solutions[i].attributes["distances_"] = distances

        solutions.sort(key=cmp_to_key(compare))

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("knn_density", lowest_is_best=False)


class NCLDensityEstimator(DensityEstimator[List[S]]):
    # todo 基于负相关学习NCL的计算方法
    pass

