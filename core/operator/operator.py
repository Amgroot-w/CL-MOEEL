from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

S = TypeVar('S')
R = TypeVar('R')


class Operator(Generic[S, R], ABC):
    """ 
    进化算子父类
        注意：所有的进化算子都要定义好其执行时所需的父代个体数以及产生的子代个体数!
    """

    @abstractmethod
    def execute(self, source: S) -> R:
        """ 执行 """
        pass

    @abstractmethod
    def get_number_of_parents(self) -> int:
        """ 该进化算子操作的父代个体数 """
        pass

    @abstractmethod
    def get_number_of_children(self) -> int:
        """ 该进化算子操作的子代个体数 """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """ 算子名称 """
        pass

# 装饰器：检查进化算子的概率是否合法（0~1之间）
def check_valid_probability_value(func):
    def func_wrapper(self, probability: float):
        if probability > 1.0:
            raise Exception('The probability is greater than one: {}'.format(probability))
        elif probability < 0.0:
            raise Exception('The probability is lower than zero: {}'.format(probability))

        res = func(self, probability)
        return res
    return func_wrapper


class Selection(Operator[S, R], ABC):
    """ 选择算子 """

    def __init__(self):
        pass  # 选择算子无需概率，因此没有进行函数装饰


class Crossover(Operator[List[S], List[R]], ABC):
    """ 交叉算子 """

    @check_valid_probability_value
    def __init__(self, probability: float):
        self.probability = probability  # 交叉概率


class Mutation(Operator[S, S], ABC):
    """ 变异算子 """

    @check_valid_probability_value
    def __init__(self, probability: float):
        self.probability = probability  # 变异概率



