from abc import abstractmethod
from functools import cmp_to_key
from turtle import dot
from typing import List, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.util.comparator import (SolutionAttributeComparator,
                                 SolutionObjectiveComparator)

S = TypeVar('S')

class SolutionSelector:
    """ 
    选解-父类

        注意：初始化时须传入PF，而不是Pop
    """

    def __init__(self, front: List[S]):
        self.front = front  # Pareto前沿
        self.selected_solution = None  # 被选择的解

    @abstractmethod
    def execute(self) -> S:
        """ 选解 """
        pass

    @abstractmethod
    def visualize(self):
        """ 可视化选解结果 """
        pass
    
    @staticmethod
    def get_name(self) -> str:
        """ 方法名 """
        pass


class BestObjectiveSelector(SolutionSelector):
    """ 最优目标函数选择法 """

    def __init__(self, front: List[S], obj: str = 'RMSE', lowest_is_best: bool = True):
        super().__init__(front)

        if obj == 'RMSE':
            obj_index = 0 
        elif obj == 'Diversity':
            obj_index = 1
        elif obj == 'Complexity':
            obj_index = 2
        else:
            raise AttributeError(f'Input obj is invalid: {obj}')

        self.obj = obj
        self.comparator = SolutionObjectiveComparator(obj_index, lowest_is_best)

    def execute(self) -> S:
        self.front.sort(key=cmp_to_key(self.comparator.compare))  # 从小到大排序
        self.selected_solution = self.front[0]  # 返回目标函数值最小的解
        return self.selected_solution
    
    def visualize(self, save_path=None, show=False, ax=None):
        pop = []
        for s in self.front:
            pop.append(s.objectives)
        pop = pd.DataFrame(pop)  # 获取所有点    

        p = self.selected_solution.objectives  # 被选择的点

        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pop.iloc[:, 0], pop.iloc[:, 1], pop.iloc[:, 2], 
                   c='yellow', s=60, alpha=0.7, linewidth=0.5, edgecolors='black', label='Non-dominated solution')  # 画出所有点
                   
        ax.scatter(p[0], p[1], p[2], 
                   c='red', s=300, marker='*', alpha=1.0, linewidth=0.5, edgecolors='red', label='Selected solution: Best RMSE')  # 画出被选择的点

        ax.view_init(elev=40, azim=40)  # elev仰角, azim方位角
        ax.set_xlabel('RMSE')  # x轴名称: RMSE
        ax.set_ylabel('Diversity')  # y轴名称: Diversity
        ax.set_zlabel('Complexity')  # z轴名称: Complexity
        
        # plt.title(f'Select solution: Best {self.obj}')
        plt.legend(loc='right', bbox_to_anchor=(1.0, 0.8))

        if save_path is not None:
            plt.savefig(f'{save_path}/SelectSolution_{self.get_name()}.jpg', dpi=600, bbox_inches='tight')

        if show:
            plt.show()

    def get_name(self):
        return f'Best{self.obj}'


class SecondBestObjectiveSelector(BestObjectiveSelector):
    """ 次优目标函数选择法 """

    def __init__(self, front: List[S], obj: str = 'RMSE', lowest_is_best: bool = True):
        super().__init__(front, obj, lowest_is_best)
    
    def execute(self, order_index: int = 1) -> S:
        self.front.sort(key=cmp_to_key(self.comparator.compare))  # 从小到大排序
        self.selected_solution = self.front[order_index]  # 返回排序后第order_index个解
        return self.selected_solution


class KneePointSelector(SolutionSelector):
    """ 膝点法 """

    def __init__(self, front: List[S]):
        super().__init__(front)

        self.comparator = SolutionAttributeComparator(key='knee_distance', lowest_is_best=False)  # 基于属性'knee_distance'的比较器

        # 确定极平面：PF面上3个目标函数最大值对应的3个点确定的平面
        self.s_max_obj1 = BestObjectiveSelector(front=front, obj='RMSE', lowest_is_best=False).execute()
        self.s_max_obj2 = BestObjectiveSelector(front=front, obj='Diversity', lowest_is_best=False).execute()
        self.s_max_obj3 = BestObjectiveSelector(front=front, obj='Complexity', lowest_is_best=False).execute()

        # 处理构成极平面的三点有重复的异常情况
        a, b, c = self.s_max_obj1.objectives, self.s_max_obj2.objectives, self.s_max_obj3.objectives
        if a == b:
            if b == c:
                # abc相同
                self.s_max_obj3 = SecondBestObjectiveSelector(
                    front=front, obj='Complexity', lowest_is_best=False).execute()
                self.s_max_obj2 = SecondBestObjectiveSelector(
                    front=front, obj='Diversity', lowest_is_best=False).execute()
                # 如果各自的第二小的目标函数也相同，则选择第三小的点    
                if self.s_max_obj2.objectives == self.s_max_obj3.objectives:
                    self.s_max_obj3 = SecondBestObjectiveSelector(
                        front=front, obj='Complexity', lowest_is_best=False).execute(order_index=2)
            else:
                # ab相同
                self.s_max_obj1 = SecondBestObjectiveSelector(
                    front=front, obj='RMSE', lowest_is_best=False).execute()
        elif a == c:
            # ac相同
            self.s_max_obj3 = SecondBestObjectiveSelector(
                front=front, obj='Complexity', lowest_is_best=False).execute()
        elif b == c:
            # bc相同
            self.s_max_obj3 = SecondBestObjectiveSelector(
                front=front, obj='Complexity', lowest_is_best=False).execute()
        else:
            pass

        # 为每个解的'knee_distance'属性赋值
        dot2plane = dot2planeDistance(A=self.s_max_obj1.objectives, B=self.s_max_obj2.objectives, C=self.s_max_obj3.objectives)
        for NonDominatedSolution in front:
            # 判断P点与平面的相对位置（定义点在平面下方时距离为正，因此在上方为-1、下方为+1、位于平面上为0）
            sign = -1 * dot2plane.decide(P=NonDominatedSolution.objectives)
            # 计算距离的值并加上符号
            NonDominatedSolution.attributes['knee_distance'] = sign * dot2plane.compute(P=NonDominatedSolution.objectives)

    def execute(self):
        self.front.sort(key=cmp_to_key(self.comparator.compare))  # 从大到小排序
        self.selected_solution = self.front[0]  # 选择距离最大的解
        return self.selected_solution
    
    def visualize(self, save_path=None, show=False, ax=None):
        pop = []
        for s in self.front:
            pop.append(s.objectives)
        pop = pd.DataFrame(pop)  # 获取所有点    

        p, a, b, c = self.selected_solution.objectives, self.s_max_obj1.objectives, self.s_max_obj2.objectives, self.s_max_obj3.objectives  # 四个点
        plane = pd.DataFrame([a, b, c, a])  # 构成平面的三个点，最后并上第一个点，便于画图

        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pop.iloc[:, 0], pop.iloc[:, 1], pop.iloc[:, 2], 
                   c='yellow', s=60, alpha=0.7, linewidth=0.5, edgecolors='black', label='Non-dominated solution')  # 画出所有点

        ax.scatter(plane.iloc[:, 0], plane.iloc[:, 1], plane.iloc[:, 2], 
                   c='red', s=60, alpha=0.7, linewidth=0.5, edgecolors='black', label='Polar solution')  # 画出构成平面的三个点

        ax.scatter(p[0], p[1], p[2], 
                   c='blue', s=150, marker='^', alpha=1.0, linewidth=0.5, edgecolors='blue', label='Selected solution: Knee-point')  # 画出被选择的点

        ax.plot(plane.iloc[:, 0], plane.iloc[:, 1], plane.iloc[:, 2], '-r', label='Polar plane')  # 用实线画出极平面

        ax.view_init(elev=40, azim=40)  # elev仰角, azim方位角
        ax.set_xlabel('RMSE')  # x轴名称: RMSE
        ax.set_ylabel('Diversity')  # y轴名称: Diversity
        ax.set_zlabel('Complexity')  # z轴名称: Complexity
        
        # plt.title(f'Select solution: Knee-point')
        plt.legend(loc='right', bbox_to_anchor=(1.0, 0.8))

        if save_path is not None:
            plt.savefig(f'{save_path}/SelectSolution_{self.get_name()}.jpg', dpi=600, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def get_name(self):
        return 'KneePoint'


class dot2planeDistance:
    """
    计算点到平面的距离

        平面方程: Ax + By + Cz + D = 0
    """
    def __init__(self, A, B, C):
        # A, B, C为构成平面的三个点的坐标（含有3个元素的列表）
        x1, y1, z1 = A[0], A[1], A[2]
        x2, y2, z2 = B[0], B[1], B[2]
        x3, y3, z3 = C[0], C[1], C[2]
        self.A = (y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 -z1)
        self.B = (z2 - z1)*(x3 - x1) - (z3 - z1)*(x2 - x1)
        self.C = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
        self.D = -(self.A * x1 + self.B * y1 + self.C * z1)
        self.sqrt = (self.A**2 + self.B**2 + self.C**2) ** 0.5

    def compute(self, P):
        x, y, z = P[0], P[1], P[2]
        return abs(self.A * x + self.B * y + self.C * z + self.D) / self.sqrt
    
    def decide(self, P):
        x, y, z = P[0], P[1], P[2]
        return np.sign(self.A * x + self.B * y + self.C * z + self.D)

