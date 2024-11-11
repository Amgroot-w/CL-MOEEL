import time
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from core.util.comparator import Comparator, DominanceComparator

S = TypeVar('S')

def get_non_dominated_solutions(solutions: List[S]) -> List[S]:
    archive: Archive = NonDominatedSolutionsArchive()

    for solution in solutions:
        archive.add(solution)

    return archive.solution_list

class Archive(Generic[S], ABC):

    def __init__(self):
        self.solution_list: List[S] = []

    @abstractmethod
    def add(self, solution: S) -> bool:
        pass

    def get(self, index: int) -> S:
        return self.solution_list[index]

    def size(self) -> int:
        return len(self.solution_list)

    def get_name(self) -> str:
        return self.__class__.__name__


class NonDominatedSolutionsArchive(Archive[S]):

    def __init__(self, dominance_comparator: Comparator = DominanceComparator()):
        super(NonDominatedSolutionsArchive, self).__init__()
        self.comparator = dominance_comparator

    def add(self, solution: S) -> bool:
        is_dominated = False
        is_contained = False

        if len(self.solution_list) == 0:
            self.solution_list.append(solution)
            return True
        else:
            number_of_deleted_solutions = 0

            # New copy of list and enumerate
            for index, current_solution in enumerate(list(self.solution_list)):
                is_dominated_flag = self.comparator.compare(solution, current_solution)
                if is_dominated_flag == -1:
                    del self.solution_list[index - number_of_deleted_solutions]
                    number_of_deleted_solutions += 1
                elif is_dominated_flag == 1:
                    is_dominated = True
                    break
                elif is_dominated_flag == 0:
                    if solution.objectives == current_solution.objectives:
                        is_contained = True
                        break

        if not is_dominated and not is_contained:
            self.solution_list.append(solution)
            return True

        return False
                

class FindNDSetMethod(object):
    def __init__(self):
        self.obj_num = None  # 目标维数
        self.NDSet = None    # 非支配解集（np数组）
        self.runtime = None  # 算法运行时间
        self.name = None     # 算法名称
        self.print = False    # 是否打印输出

    # 比较两个解(p和q)的支配关系
    @staticmethod
    def compare(p, q):
        """
        该compare方法不限制目标函数维数，适用于n维；

        :param 具有相同shape的n维向量p、q；
        :return 0表示p和q非支配，1表示p支配q，2表示q支配p；

        vec变量与p、q的shape一致；
        vec中的元素表示p和q中对应位置元素的大小比较结果：1表示小于、0表示等于、-1表示大于；
        ***注意：此处“小于”表示目标函数更优，即优化算法目标为：min f(x)；
        p等于q时，返回的结果是”p、q非支配“；

                            支配关系的判断逻辑表
        --------------------------------------------------------
            vec中所含元素种类     p支配q     q支配p     p、q非支配
        --------------------------------------------------------
                1                √
                0                                      √
                -1                         √
              1, 0               √
              1, -1                                    √
              0, -1                        √
             1, 0, -1                                  √
        --------------------------------------------------------
        """
        # 在元素尺度（即各个目标函数维度）比较p和q的大小
        vec = np.piecewise(p, [p < q, p == q, p > q], [1, 0, -1])
        # 提取vec中的元素种类（可能有三种：1、0、-1）
        vec_elements = np.unique(vec)
        # 条件判断（判断逻辑见上面的表）
        if np.isin(1, vec_elements):
            if np.isin(-1, vec_elements):
                return 0  # p、q非支配
            else:
                return 1  # p支配q
        else:
            if np.isin(-1, vec_elements):
                return 2  # q支配p
            else:
                return 0  # p、q非支配

    # 算法流程（各子类重构此方法）
    def findNDSet(self, *args):
        pass

    # 运行算法
    def run(self):
        start = time.time()
        self.findNDSet()
        end = time.time()
        self.runtime = end - start  # 记录算法运行时间
        if self.print:
            print('Algorithm: %s\tRuntime：%.6fs' % (self.name, self.runtime))

    # 可视化非支配解的分布
    def visualize_NDSet(self, show=True):
        # 目标函数为二维
        if self.obj_num == 2:
            plt.scatter(self.pop[:, 0], self.pop[:, 1], c='blue', s=65,
                        alpha=0.5, label='dominated solution')
            plt.scatter(self.NDSet[:, 0], self.NDSet[:, 1], c='red', s=65,
                        alpha=0.8, label='non-dominated solution')
            plt.legend(loc='upper right')
            plt.title('Method: %s' % self.name)
            if show:
                plt.show()
        # 目标函数为三维
        elif self.obj_num == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.pop[:, 0], self.pop[:, 1], self.pop[:, 2], c='blue', s=65,
                       alpha=0.5, label='dominated solution')
            ax.scatter(self.NDSet[:, 0], self.NDSet[:, 1], self.NDSet[:, 2], c='red', s=65,
                       alpha=1, edgecolors='k', label='non-dominated solution')
            ax.view_init(elev=20, azim=300)  # 分别表示：仰角、方位角
            plt.legend(loc='upper right')
            plt.title('Method: %s' % self.name)
            if show:
                plt.show()
        else:
            print('*** ERROR: Only 2D or 3D population can be visualized!')

