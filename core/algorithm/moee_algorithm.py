import copy
import time
from abc import abstractmethod
from typing import List

import pandas as pd
from rich.progress import track

from core.operator import Crossover, Mutation, Selection
from core.problem import MOEEProblem, MOEESolution
from core.util.checking import Check
from core.util.clustering import PopulationCluster
from core.util.initialization import Initializer
from core.util.recording import Recorder
from core.util.density_estimator import CrowdingDistance, NCLDensityEstimator

"""
MOEE算法父类
"""

class MOEE_Algorithm:
    """ 
    多目标进化集成(MOEE)算法-父类
        定义了MOEE算法的基本流程
    
    数据类型：
        - problem:    MOEEProblem
        - solution:   MOEESolution
        - population: List[MOEESolution]
    """

    def __init__(self,
                 problem: MOEEProblem,  # 问题
                 selection: Selection,  # 选择算子
                 crossover: Crossover,  # 交叉算子
                 mutation: Mutation,    # 变异算子
                 population_size: int,  # 种群大小
                 offspring_population_size: int,  # 子种群大小
                 max_epoch: int,  # 进化代数
                 initializer: Initializer):  # 种群初始化器
        self.problem = problem
        self.selection_operator = selection
        self.crossover_operator = crossover
        self.mutation_operator = mutation
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.max_epoch = max_epoch
        self.initializer = initializer
        self.recorder = Recorder()  # 进化过程记录器

    def run(self, running_info: str = 'Running:'):
        """ 执行优化算法 """
        # 开始计时
        start = time.time()
        # 产生初始种群
        self.pop = self.create_initial_population()
        # 评价初始种群
        self.pop = self.evaluate_population(self.pop)
        # 初始化进化过程
        self.init_progress()
        # 记录初始种群
        self.recorder.record_pop(population=self.pop, epoch=0)
        # 进化
        for self.epoch in track(range(self.max_epoch), description=running_info):
            # 执行一次进化操作
            self.step()
            # 记录进化过程
            self.recorder.record_pop(population=self.pop, epoch=self.epoch+1)
        # 计时结束
        self.runtime = time.time() - start

    def init_progress(self) -> None:
        """ 初始化进化过程 """
        # 基类的init_progress()方法默认没有任何操作，子类可以不用重写此方法，表示和基类一样无初始化进程操作；
        # 当子类需要初始化进程操作时，则重写该方法，但是不能返回任何值。
        pass

    def create_initial_population(self) -> List[MOEESolution]:
        """ 产生初始种群 """
        # 调用传入的种群初始化器，产生初始种群
        initial_population = self.initializer.create()
        return initial_population

    def evaluate_population(self, population: List[MOEESolution]) -> List[MOEESolution]:
        """ 评价种群 """
        self.problem.evaluate(population)
        return population    

    def step(self):
        """ 进化过程 """
        # 深拷贝当前进化迭代的父代种群
        parents_population = copy.deepcopy(self.pop)
        # 选择：产生父代种群
        mating_population = self.selection(self.pop)
        # 交叉/变异：产生子代种群
        offspring_population = self.reproduction(mating_population)
        # 评价子代种群：计算种群中各解的三个目标函数
        offspring_population = self.evaluate_population(offspring_population)
        # 环境选择：产生最终的种群进入下一次进化
        self.pop = self.replacement(parents_population, offspring_population)

    @abstractmethod
    def selection(self, population: List[MOEESolution]) -> List[MOEESolution]:
        """ 选择 """
        pass

    @abstractmethod
    def reproduction(self, mating_population: List[MOEESolution]) -> List[MOEESolution]:
        """ 交叉 & 变异 """
        pass

    @abstractmethod
    def replacement(self, population: List[MOEESolution], offspring_population: List[MOEESolution]) -> List[
        List[MOEESolution]]:
        """ 环境选择 """
        pass

    def get_result(self) -> List[MOEESolution]:
        """ 获取算法运行结果：进化完成后的种群 """
        return self.pop

    def get_log(self) -> dict:
        """ 获取记录的算法信息 """
        log = {
            'algorithm': self.get_name(),  # 算法名
            'problem': self.problem.get_name(),  # 问题名
            'dataset': self.problem.dataset_name,  # 数据集名
            'meta_model': self.problem.meta_model_selection,  # meta模型选择
            'div_method': self.problem.div_method,  # 分散性指标计算方法
            'runtime': self.runtime,  # 运行时间
            'pop': self.recorder.get_log_pop(),  # 所有进化代数的种群分布
            'cluster_info': self.recorder.get_log_cluster(),  # 每个聚类子种群的优势目标函数
            'strategy_p': self.recorder.get_log_strategy_p(),  # 进化过程中三个变长策略的选择概率的变化情况
            'selection_operator': self.selection_operator.get_name(), # 选择算子
            'crossover_operator': self.crossover_operator.get_name(), # 交叉算子
            'mutation_operator': self.mutation_operator.get_name(), # 变异算子
            'population_size': self.population_size,  # 种群大小
            'max_epoch': self.max_epoch,  # 最大迭代次数
        }
        return log

    @abstractmethod
    def get_name(self) -> str:
        """ 算法名 """
        pass

    @property
    def label(self) -> str:
        return f'{self.get_name()}.{self.problem.get_name()}'


class MOEEC_Algorithm(MOEE_Algorithm):
    """
    多目标进化集成聚类(MOEEC)算法-父类
    
    与父类MOEE_Algorithm的区别：
        - 在step()中增加了第一步的cluster过程；
        - 定义了cluster()函数，实现聚类过程（即为每个个体赋予一个聚类标签属性）；
    
    其他的复杂过程需要子类在继承时实现；
    """

    def __init__(self,
                 problem: MOEEProblem,  # 问题
                 selection: Selection,  # 选择算子
                 crossover: Crossover,  # 交叉算子
                 mutation: Mutation,    # 变异算子
                 population_size: int,  # 种群大小
                 offspring_population_size: int,  # 子种群大小
                 max_epoch: int,  # 进化代数
                 initializer: Initializer):  # 种群初始化器
        super(MOEEC_Algorithm, self).__init__(
            problem=problem,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            max_epoch=max_epoch,
            initializer=initializer)
        
        self.clusteror = PopulationCluster(n_clusters=3)  # 初始化聚类器
        
    def step(self):
        """ 重写进化过程 """    
        # 聚类：将整个种群聚为k类
        self.cluster(self.pop)
        # 深拷贝当前进化迭代的父代种群
        parents_population = copy.deepcopy(self.pop)    
        # 选择：产生父代种群
        mating_population = self.selection(self.pop)
        # 交叉/变异：产生子代种群
        offspring_population = self.reproduction(mating_population)
        # 评价子代种群：计算种群中各解的三个目标函数
        offspring_population = self.evaluate_population(offspring_population)
        # 环境选择：产生最终的种群进入下一次进化
        self.pop = self.replacement(parents_population, offspring_population)

    def cluster(self, population: List[MOEESolution]):
        """ 聚类 """
        # 注意：此操作没有返回值，而是为pop中的每个个体解计算一个新的属性: 聚类标签/cluster_label
        
        # 种群大小必须大于3才能聚类，因为聚类类别数当前默认设置为3
        Check.that(len(population) >= 3, f'The minimal population size is 3, but input {len(population)}')
        
        pop = pd.DataFrame(
            [solution.objectives for solution in population], 
            columns=['RMSE', 'Diversity', 'Complexity'])

        labels = self.clusteror.cluster(pop)  # 聚类，并获取聚类标签
        
        for solution, label in zip(population, labels):
            solution.attributes['cluster_label'] = label  # 为种群个体的聚类标签cluster_label属性赋值
        
