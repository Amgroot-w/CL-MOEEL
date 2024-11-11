from abc import ABC, abstractmethod
from typing import List


class AggregativeFunction(ABC):
    """ 聚合函数 """

    @abstractmethod
    def compute(self, vector: List[float], weight_vector: List[float]) -> float:
        pass

    @abstractmethod
    def update(self, vector: List[float]) -> None:
        pass


class WeightedSum(AggregativeFunction):
    """ 权重求和法 """

    def compute(self, vector: List[float], weight_vector: List[float]) -> float:
        """ 计算聚合函数值 """
        return sum(map(lambda x, y: x * y, vector, weight_vector))

    def update(self, vector: List[float]) -> None:
        """ 更新（此方法无操作） """
        pass


class Tschebycheff(AggregativeFunction):
    """ 切比雪夫聚合 """

    def __init__(self, dimension: int):
        # 聚合函数对象初始化的时候，也初始化了理想点（每一维度均初始化为inf）
        self.ideal_point = IdealPoint(dimension)  # 理想点

    def compute(self, vector: List[float], weight_vector: List[float]) -> float:
        """ 计算聚合函数值 """
        max_fun = -1.0e+30

        for i in range(len(vector)):
            diff = abs(vector[i] - self.ideal_point.point[i])

            if weight_vector[i] == 0:
                feval = 0.0001 * diff
            else:
                feval = diff * weight_vector[i]

            if feval > max_fun:
                max_fun = feval

        return max_fun

    def update(self, vector: List[float]) -> None:
        """ 更新理想点 """
        # 传入的vector应该对应solution的objectives变量（list型）
        self.ideal_point.update(vector)


class Point(ABC):

    @abstractmethod
    def update(self, vector: List[float]) -> None:
        pass


class IdealPoint(Point):
    """ 理想点 """

    def __init__(self, dimension: int):
        self.point = dimension * [float("inf")]  # 初始化理想点为inf

    def update(self, vector: List[float]) -> None:
        """ 更新理想点 """
        # 默认理想点为各维目标函数的最小值
        self.point = [y if x > y else x for x, y in zip(self.point, vector)]

