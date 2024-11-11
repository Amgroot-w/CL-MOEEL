from abc import abstractmethod
from functools import cmp_to_key
from typing import List, TypeVar

from core.operator import SlackBinaryTournamentSelection
from core.util.comparator import SolutionObjectiveComparator
from core.util.density_estimator import CrowdingDistance
from core.util.ranking import FastNonDominatedRanking

S = TypeVar('S')

class Replacement:
    """ 环境选择 """
    @abstractmethod
    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """
        输入：父代种群、子代种群
        输出：环境选择后的种群
        """
        pass


class ObjectiveBasedSlackBinaryTournamentReplacement(Replacement):
    """
    基于目标函数值和二元锦标赛的环境选择 (EvoCNN)
        基于第一维目标函数值（RMSE）的精英保留 + 松弛二元锦标赛补充选择剩余个体
    """

    def __init__(self, elite_ratio: float):
        self.elite_ratio = elite_ratio  # 精英个体的保留比例
        self.comparator = SolutionObjectiveComparator(obj_index=0, lowest_is_best=True)  # 基于第1维目标函数（RMSE）的比较器
        self.SlackBTS = SlackBinaryTournamentSelection()  # 松弛二元锦标赛选择

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        # 计算要保留的精英个体数目
        elite_num = round(self.elite_ratio * len(solution_list))
        # 合并父代种群和子代种群
        join_population = solution_list + offspring_list
        # 将合并后的种群按照第1维目标函数值（RMSE）进行排序
        join_population.sort(key=cmp_to_key(self.comparator.compare))
        # 选择指定数目的精英个体直接进入下一代种群
        result_list = join_population[:elite_num]
        # 用松弛二元锦标赛选择出剩余的个体补充进入下一代种群
        while len(result_list) < len(solution_list):
            solution = self.SlackBTS.execute(join_population[elite_num:])
            result_list.append(solution)
        return result_list


class RankingAndDensityEstimatorReplacement(Replacement):
    """ 
    基于Rank和拥挤距离的环境选择 (NSGA-II) 
        基于支配关系的精英保留 + 基于拥挤距离补充选择剩余个体

    截断策略：
        - SEQUENTIAL
        - ONE_SHOT
    """
    
    def __init__(self, 
        ranking = FastNonDominatedRanking(),  # 快速非支配排序
        density_estimator = CrowdingDistance(),  # 拥挤距离算子
        removal_policy='ONE_SHOT'  # 截断策略
    ):
        self.ranking = ranking
        self.density_estimator = density_estimator
        self.removal_policy = removal_policy

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """
        该函数的作用是：输入当前种群和经过进化算子操作之后产生的种群，输出经过replace后的种群（种群大小与当前种群大小一致）
        """
        # 先将原种群和产生的新种群合并
        join_population = solution_list + offspring_list
        # 然后计算合并种群的rank（计算得到的分层结果保存在self中）
        self.ranking.compute_ranking(join_population)  # 这个不需要返回值，因为计算结果已经保存在self.ranked_sublists里面了
        # 选择截断策略（两种，一种是逐步地，一种是一次性的）
        if self.removal_policy == 'ONE_SHOT':
            result_list = self.sequential_truncation(0, len(solution_list))
        elif self.removal_policy == 'SEQUENTIAL':
            result_list = self.one_shot_truncation(0, len(solution_list))
        else:
            raise AttributeError('removal_policy must be SEQUENTIAL or ONE_SHOT !')
        # 上述步骤完成后，即可得到满足个数要求的replace之后的子代种群
        return result_list

    # 截断策略1：按照拥挤距离排序，逐步地删除直到满足要求（sequential: 需要逐个删除，每删除一个都要重新计算一次拥挤距离）
    def sequential_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
        current_ranked_solutions = self.ranking.get_subfront(ranking_id)  # 首先获取第0层的分层结果（list型，储存所有solution）
        self.density_estimator.compute_density_estimator(current_ranked_solutions)  # 为所有solution计算拥挤距离

        result_list: List[S] = []
        # 当前层的所有个体数不足以提供子代要求的个体数量，那么就从下一层补充（通过递归调用的方法，知道达到子代种群数量要求）
        if len(current_ranked_solutions) < size_of_the_result_list:
            # 先将当前的种群并入结果中
            result_list.extend(self.ranking.get_subfront(ranking_id))
            # 由于数量不够，采用递归调用的方法从下一层中补充个体
            result_list.extend(self.sequential_truncation(ranking_id + 1, size_of_the_result_list - len(
                current_ranked_solutions)))

        # 当前层的所有个体能够提供子代要求的所有个体，那么需要删除不必要的个体，只保留指定数量的个体
        else:
            # 首先将当前的所有个体并入结果中
            for solution in current_ranked_solutions:
                result_list.append(solution)
            # 然后开始删除多余的个体
            while len(result_list) > size_of_the_result_list:
                # 按照拥挤距离排序，拥挤距离大的拍的靠前
                self.density_estimator.sort(result_list)
                # 删除排在最后面的个体（即拥挤距离最小的个体，它的分散性最不好）
                del result_list[-1]
                # 由于种群中任何一个个体的变化，都会导致种群中所有个体的拥挤距离指标改变（这也是NSGA-II算法缺点之一，导致算法时间成本增加），
                # 因此此处在删除一个个体之后，需要重新计算一次拥挤距离指标
                self.density_estimator.compute_density_estimator(result_list)

        return result_list

    # 截断策略2：按照拥挤距离排序，之后一次性获取满足要求的个体（one_shot: 拥挤距离只计算一次，用计算好的拥挤距离直接选取足够量的个体）
    def one_shot_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
        current_ranked_solutions = self.ranking.get_subfront(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)

        result_list: List[S] = []

        # 该情况下，该策略与上一个策略思想一致
        # 当前层的所有个体数不足以提供子代要求的个体数量，那么就从下一层补充（通过递归调用的方法，知道达到子代种群数量要求）
        if len(current_ranked_solutions) < size_of_the_result_list:
            result_list.extend(self.ranking.get_subfront(ranking_id))
            result_list.extend(self.one_shot_truncation(ranking_id + 1, size_of_the_result_list - len(
                current_ranked_solutions)))

        # 两个截断策略主要体现在此处：
        # 若当前层的所有个体能够提供子代要求的所有个体，则逐一的将排序后的个体并入结果中
        else:
            self.density_estimator.sort(current_ranked_solutions)  # 根据拥挤距离排序
            i = 0
            while len(result_list) < size_of_the_result_list:
                result_list.append(current_ranked_solutions[i])  # 逐一的将排序后的个体并入结果中
                i += 1

        return result_list


