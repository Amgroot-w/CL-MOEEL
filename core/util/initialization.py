import os
import pickle
from abc import abstractmethod
from typing import List, TypeVar

from core.util.checking import Check

S = TypeVar('S')

class Initializer:
    """ 种群初始化器 """
    def __init__(self, population_size: int):
        self.population_size = population_size
    
    @abstractmethod
    def create(self) -> List[S]:
        pass


class RandomInitializer(Initializer):
    """ 随机初始化 """

    def __init__(self, population_size, problem):
        super(RandomInitializer, self).__init__(population_size)
        self.problem = problem
    
    @abstractmethod
    def create(self) -> List[S]:
        # 调用Problem类的creat_solution()方法，产生一个随机初始化的种群
        initial_population = [
            self.problem.create_solution()
            for _ in range(self.population_size)
        ]
        return initial_population


class LoadFromFileInitializer(Initializer):
    """ 从保存的文件中读取初始化种群 """

    def __init__(self, population_size, file_path):
        super(LoadFromFileInitializer, self).__init__(population_size)
        self.file_path = file_path  # 种群文件路径，最后一级必须是文件（即：../../xxx.pkl）

    def create(self) -> List[S]:
        # 该方法用pickle读取预先保存的.pkl文件，该文件必须同样地由pickle保存而得到
        # 该预保存文件的内容，必须是List[MOEESolution]类型
        with open(self.file_path, 'rb') as f:
            initial_population = pickle.load(f)
        
        Check.that(
            len(initial_population) == self.population_size,
            f'Population size {self.population_size} is not equal to {len(initial_population)} which is pre-defined in file: {self.file_path}'
        )  # 检查参数population_size是否与文件中的种群大小是一致的，如果不一致则抛出异常
        
        print('Initialize the population using pre-defined solutions in file: {}'.format(self.file_path))

        return initial_population










