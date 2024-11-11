import random
from abc import abstractmethod

import numpy as np

from core.util.evolving_level import (EpochBasedEvolvingLevel,
                                     PFDistanceBasedEvolvingLevel)

"""
定义了MOEE中涉及到的策略决定器
"""

class StrategyDecider:
    """ 决策决定器 """

    @abstractmethod
    def execute(self):
        pass


class DecideVariationStrategyByRandom(StrategyDecider):
    """ 决定变长策略#1：随机确定 """
    
    def execute(self):
        rand = random.random()  # 产生0~1范围的随机数
        if 0 <= rand < 1/3:
            return 'variation1'
        elif 1/3 <= rand < 2/3:
            return 'variation2'
        else:
            return 'fixed'


class DecideVariationStrategyByEpoch(StrategyDecider):
    """
    决定变长策略#2：通过Epoch确定
        - 进化前期为了保证分散性，采用波动程度较小的交叉策略；
        - 进化后期由于大量的解都处于一个进化停滞状态，所以应该采用波动程度更大的交叉策略；
    """

    def __init__(self, max_epoch: int):
        self.evolving_level_computer = EpochBasedEvolvingLevel(max_epoch=max_epoch)  # 基于Epoch的进化程度计算器
    
    def execute(self, epoch: int):
        level = self.evolving_level_computer.compute(epoch=epoch)  # 计算当前的进化程度（值域范围0~1）
        if 0 <= level < 1/3:
            return 'fixed'  # 进化初期
        elif 1/3 <= level < 2/3:
            return 'variation2'  # 进化中期
        else:
            return 'variation1'  # 进化末期
 

class DecideVariationStrategyByPFdistance(StrategyDecider):
    """ 决定变长策略#3：通过相邻两次进化的PF距离确定 """

    def __init__(self):
        self.evolving_level_computer = PFDistanceBasedEvolvingLevel()  # 基于PF距离的进化程度计算器
    
    def execute(self):
        pass


class DecideVariationStrategyBySelfAdaptive(StrategyDecider):
    """ 决定变长策略#4：自适应确定变长策略（SaDE论文思想） """

    def __init__(self, learning_period: int):
        self.LP = learning_period  # 学习周期
        self.success_memory = []  # 成功矩阵
        self.fail_memory = []  # 失败矩阵
        self.p1, self.p2, self.p3 = 1/3, 1/3, 1/3  # 三个策略的选择概率，均初始化为1/3

    def update_memory(self, pop1, pop2):
        """
        维护两个memory
            - pop1: 新生成的子代种群;
            - pop2: 经过自然选择后进入下一代的种群;
            - ns1, ns2, ns3: 表示三个策略的成功个数；
            - nf1, nf2, nf3: 表示三个策略的失败个数；
        """
        # 整理两个种群中的个体解保存的策略信息，得到策略的成功、失败个数
        n1 = {'variation1': 0, 'variation2': 0, 'fixed': 0}
        for s in pop1:
            if s.attributes['strategy_flag'] == 1:
                n1[s.attributes['strategy']] += 1  # 统计子种群中每个策略生成的子代个体数
                
        n2 = {'variation1': 0, 'variation2': 0, 'fixed': 0}
        for s in pop2:
            if s.attributes['strategy_flag'] == 1:
                n2[s.attributes['strategy']] += 1  # 统计子种群中每个策略生成的子代个体数
        
        # 修正：由于环境选择会有可能对同一个解选择多次，因此可能会出现策略成功数比总数还多的情况，这时让策略的成功数等于总数
        for key in n1.keys():
            if n2[key] > n1[key]:
                n2[key] = n1[key]
        
        # 根据上述统计信息计算下面6个值（三个策略的成功个数、失败个数）
        ns1, ns2, ns3 = np.array(list(n2.values()))
        nf1, nf2, nf3 = np.array(list(n1.values())) - np.array(list(n2.values()))

        # 更新两个memory
        if len(self.success_memory) < self.LP:
            # 当前进化总代数未达到学习周期数，则不更新p（对应进化初期的LP个epoch）
            self.success_memory.append([ns1, ns2, ns3])  # 添加一条策略成功信息
            self.fail_memory.append([nf1, nf2, nf3])  # 添加一条策略失败信息

        else:
            # 当前进化总代数已达到学习周期数，则更新两个memory，并且更新p（对应LP次迭代之后的每个epoch）
            del self.success_memory[0]  # 删除success_memory中的第一条信息
            del self.fail_memory[0]  # 删除fail_memory中的第一条信息
            self.success_memory.append([ns1, ns2, ns3])  # 在success_memory末尾添加一条信息
            self.fail_memory.append([nf1, nf2, nf3])  # 在fail_memory末尾添加一条信息
            
            # 更新三个策略的选择概率
            sum_success = np.sum(self.success_memory, axis=0)  # 对success_memory的LP行求和
            sum_fail = np.sum(self.fail_memory, axis=0)  # 对fail_memory的LP行求和
            S = sum_success / (sum_success + sum_fail)  # 三个策略的成功率比例
            
            # 修正：S中可能出现None值，这是因为某一条策略没有出现过，即成功0次失败0次，于是分母为0
            for i in range(len(S)):
                if np.isnan(S[i]):
                    # 如果某条策略的选择概率为0，那么从此开始以后每个迭代均不会选择该策略，于是以后每次迭代都会算出来None值
                    # 为了让变长策略即使选择概率变成0，也有可能重新被选择，此处为它赋予一个很小的值。
                    S[i] = 1e-2
            
            P = S / sum(S)  # 归一化，得到重新计算得到的三个概率
            self.p1, self.p2, self.p3 = P[0], P[1], P[2]  # 更新p

    def execute(self):
        """ 产生决策结果 """
        rand = random.random()  # 产生0~1范围的随机数
        if 0 <= rand < self.p1:
            return 'variation1'
        elif self.p1 <= rand < (self.p1 + self.p2):
            return 'variation2'
        else:
            return 'fixed'


class CrossoverStrategyDecider(StrategyDecider):
    """ 
    自适应交叉策略决策器（SaDE论文思想） 
        - 'inside': 类内交叉
        - 'outside': 类间交叉
    """

    def __init__(self, learning_period: int):
        self.LP = learning_period  # 学习周期
        self.success_memory = []  # 成功矩阵
        self.fail_memory = []  # 失败矩阵
        self.p1, self.p2 = 1/2, 1/2  # 两个策略的选择概率，均初始化为1/2

    def update_memory(self, pop1, pop2):
        """
        维护两个memory
            - pop1: 新生成的子代种群;
            - pop2: 经过自然选择后进入下一代的种群;
            - ns1, ns2: 表示两个策略的成功个数；
            - nf1, nf2: 表示两个策略的失败个数；
        """
        # 整理两个种群中的个体解保存的策略信息，得到策略的成功、失败个数
        n1 = {'inside': 0, 'outside': 0}
        for s in pop1:
            if s.attributes['strategy_flag'] == 1:
                n1[s.attributes['strategy']] += 1  # 统计子种群中每个策略生成的子代个体数
                
        n2 = {'inside': 0, 'outside': 0}
        for s in pop2:
            if s.attributes['strategy_flag'] == 1:
                n2[s.attributes['strategy']] += 1  # 统计子种群中每个策略生成的子代个体数
        
        # 修正：由于环境选择会有可能对同一个解选择多次，因此可能会出现策略成功数比总数还多的情况，这时让策略的成功数等于总数
        for key in n1.keys():
            if n2[key] > n1[key]:
                n2[key] = n1[key]
        
        # 根据上述统计信息计算下面4个值（各略的成功个数、失败个数）
        ns1, ns2 = np.array(list(n2.values()))
        nf1, nf2 = np.array(list(n1.values())) - np.array(list(n2.values()))

        # 更新两个memory
        if len(self.success_memory) < self.LP:
            # 当前进化总代数未达到学习周期数，则不更新p（对应进化初期的LP个epoch）
            self.success_memory.append([ns1, ns2])  # 添加一条策略成功信息
            self.fail_memory.append([nf1, nf2])  # 添加一条策略失败信息

        else:
            # 当前进化总代数已达到学习周期数，则更新两个memory，并且更新p（对应LP次迭代之后的每个epoch）
            del self.success_memory[0]  # 删除success_memory中的第一条信息
            del self.fail_memory[0]  # 删除fail_memory中的第一条信息
            self.success_memory.append([ns1, ns2])  # 在success_memory末尾添加一条信息
            self.fail_memory.append([nf1, nf2])  # 在fail_memory末尾添加一条信息
            
            # 更新三个策略的选择概率
            sum_success = np.sum(self.success_memory, axis=0)  # 对success_memory的LP行求和
            sum_fail = np.sum(self.fail_memory, axis=0)  # 对fail_memory的LP行求和
            S = sum_success / (sum_success + sum_fail)  # 三个策略的成功率比例
            
            # 修正：S中可能出现None值，这是因为某一条策略没有出现过，即成功0次失败0次，于是分母为0
            for i in range(len(S)):
                if np.isnan(S[i]):
                    # 如果某条策略的选择概率为0，那么从此开始以后每个迭代均不会选择该策略，于是以后每次迭代都会算出来None值
                    # 为了让变长策略即使选择概率变成0，也有可能重新被选择，此处为它赋予一个很小的值。
                    S[i] = 1e-2
            
            P = S / sum(S)  # 归一化，得到重新计算得到的两个概率
            self.p1, self.p2 = P[0], P[1]  # 更新p

    def execute(self):
        """ 产生决策结果 """
        rand = random.random()
        if rand < self.p1:
            return 'inside'
        else:
            return 'outside'

