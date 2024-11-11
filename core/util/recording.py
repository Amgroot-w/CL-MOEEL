from re import L
from typing import List, TypeVar

import pandas as pd

S = TypeVar('S')

class Recorder:
    """
    进化过程记录器
        所有的log数据均为python字典dict：key为当前的epoch，value为记录的数据，value的类型可能是：浮点数、字典、列表、DataFrame等；

    记录内容：
        1. 种群在每个Epoch的位置（即每个个体解的三个目标函数值）
            用来可视化种群分布和Pareto前沿随epoch的变化情况，并在算法结束后利用该信息计算其他的各个指标，避免在进化过程中计算，从而节省算法执行时间；
        2. 聚类子种群在每个Epoch的优势目标函数；
        3. 三个变长策略的选择概率随着Epoch的变化情况；

    """

    def __init__(self):
        self.log_pop = {}  # 记录种群位置（个体解的三个目标函数值）
        self.log_cluster = {}  # 记录聚类信息（每个聚类子种群的优势目标函数）
        self.log_strategy_p = {}  # 记录进化过程中各策略的选择概率的变化情况
        self.log_thresholds = {}  # 记录进化过程中三个目标函数的变化情况

    def record_pop(self, population: List[S], epoch: int):
        """ 记录每一代种群中解的信息 """
        # 记录解的位置
        pop = pd.DataFrame([solution.objectives for solution in population], columns=['RMSE', 'Diversity', 'Complexity'])
        
        # 记录解的聚类标签
        pop.insert(3, 'cluster_label', [None for _ in population])
        try:
            pop_cluster_labels = [s.attributes['cluster_label'] for s in population]
            pop['cluster_label'] = pop_cluster_labels  # 在最后一列并上当前种群所有个体的聚类标签
        except KeyError:
            pass  # 若算法无聚类操作，或者当前epoch还没进行过聚类（例如刚初始化后的种群，epoch=0时），则直接pass
        
        # 记录解的子学习机个数（染色体长度）
        pop.insert(4, 'length', [s.k for s in population])

        self.log_pop[epoch] = pop
    
    def record_cluster(self, info: any, epoch: int):
        """ 记录聚类子种群的优势目标函数 """
        self.log_cluster[epoch] = info

    def record_log_strategy_p(self, p: List[float], epoch: int):
        """ 记录策略的选择概率 """
        self.log_strategy_p[epoch] = p

    def record_log_thresholds(self, thresholds: list[float], epoch: int):
        """ 记录三个目标函数的阈值变化 """
        self.log_thresholds[epoch] = thresholds
        
    def get_log_pop(self):
        return self.log_pop

    def get_log_cluster(self):
        return self.log_cluster

    def get_log_strategy_p(self):
        return self.log_strategy_p

    def get_log_thresholds(self):
        return self.log_thresholds







