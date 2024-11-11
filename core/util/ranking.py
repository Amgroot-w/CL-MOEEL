from abc import ABC, abstractmethod
from typing import List, TypeVar

from core.util.comparator import (Comparator, DominanceComparator,
                                 SolutionAttributeComparator)

S = TypeVar('S')

class Ranking(List[S], ABC):
    """ Ranking """
    def __init__(self, comparator: Comparator):
        super(Ranking, self).__init__()
        self.number_of_comparisons = 0  # 累计比较次数
        self.ranked_sublists = []  # 分层结果
        self.comparator = comparator  # 解的优劣比较器

    @abstractmethod
    def compute_ranking(self, solutions: List[S], k: int = None):
        """ 计算所有个体解的rank """
        # 该方法执行之后，分层结果会储存在self.ranked_sublists中，其他方法的调用均需要此计算结果
        pass

    def get_nondominated(self):
        """ 获取第一层（Pareto前沿）上的所有个体解 """
        return self.ranked_sublists[0]

    def get_subfront(self, rank: int):
        """ 获取分层结果中的指定层的所有个体解 """
        if rank >= len(self.ranked_sublists):
            raise Exception('Invalid rank: {0}. Max rank: {1}'.format(rank, len(self.ranked_sublists) - 1))
        return self.ranked_sublists[rank]

    def get_number_of_subfronts(self):
        """ 获取层数 """
        return len(self.ranked_sublists)

    @classmethod
    def get_comparator(cls) -> Comparator:
        """ 获取比较器 """
        pass


class FastNonDominatedRanking(Ranking[List[S]]):
    """ 快速非支配排序 (NSGA-II) """

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(FastNonDominatedRanking, self).__init__(comparator)

    # 这个函数的作用：得到分层的排序结果F（即第一层为NDSet，第二层为次之，以此类推）
    def compute_ranking(self, solutions: List[S], k: int = None):
        """ Compute ranking of solutions.

        :param solutions: 要排序的个体解的集合（种群）
        :param k: 指定返回的个体数（从最终的分层结果中依次选择）
        """
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solutions))]

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solutions))]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solutions) + 1)]

        for p in range(len(solutions) - 1):
            for q in range(p + 1, len(solutions)):
                # 调用compare方法，比较两个个体
                dominance_test_result = self.comparator.compare(solutions[p], solutions[q])
                # 记录比较次数
                self.number_of_comparisons += 1

                if dominance_test_result == -1:   # 结果为-1，表示p支配q
                    ith_dominated[p].append(q)    # 在被p支配的个体清单中加上q
                    dominating_ith[q] += 1        # 能够支配q的个体数目加1
                elif dominance_test_result == 1:  # 结果为1，表示q支配p
                    ith_dominated[q].append(p)    # 在被q支配的个体清单中加上p
                    dominating_ith[p] += 1        # 能够支配p的个体数目加1

        # 得到front[0]，即NDSet，即不被其他任何个体支配的个体的集合
        for i in range(len(solutions)):
            if dominating_ith[i] == 0:
                front[0].append(i)  # 将该非支配解的index加入NDSet中
                solutions[i].attributes['dominance_ranking'] = 0  # 将该非支配解的rank记为0（表示改解位于PF面上）

        # 得到分层之后的其他层中的个体，并为其他所有个体赋予一个层数
        i = 0
        while len(front[i]) != 0:
            i += 1
            for p in front[i - 1]:
                if p <= len(ith_dominated):
                    for q in ith_dominated[p]:
                        dominating_ith[q] -= 1
                        if dominating_ith[q] == 0:
                            front[i].append(q)
                            solutions[q].attributes['dominance_ranking'] = i

        # 得到分层结果（上面得到的分层结果中只是存储了个体的index，现在要得到存储了相应个体解的分层结果）
        self.ranked_sublists = [[]] * i
        for j in range(i):
            q = [0] * len(front[j])
            for m in range(len(front[j])):
                q[m] = solutions[front[j][m]]
            self.ranked_sublists[j] = q

        # 如果设置了返回个体解的数量上限，则只在分层结果中取出该数量的个体解作为结果返回
        if k:
            count = 0
            for i, front in enumerate(self.ranked_sublists):
                count += len(front)
                if count >= k:
                    self.ranked_sublists = self.ranked_sublists[:i + 1]
                    break

        return self.ranked_sublists

    @classmethod 
    def get_comparator(cls) -> Comparator:
        """ 获取由此非支配排序之后得到的基于支配rank的比较器 """
        # 这里的逻辑是：
        #     先实例化非支配排序Ranking类，并计算好种群中所有个体解的rank，计算过程依赖基于支配关系的比较器（比较目标函数，向量）；
        # 待计算完成后，即可为每个解赋予一个属性（即rank值，标量），通过该标量又可以构造一个比较器（该比较器的需要传入的属性名称key是
        # 和该class计算过程中赋予解的属性名称是一致的），用来后续比较解的rank值。
        return SolutionAttributeComparator('dominance_ranking')

