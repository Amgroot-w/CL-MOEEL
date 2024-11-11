import copy
import random
from typing import List, TypeVar

import pandas as pd
from cv2 import CV_32F

from core.operator import Crossover
from core.problem import MOEESolution
from core.util.checking import Check

S = TypeVar('S')


class VariableLengthCrossover(Crossover):
    
    def __init__(self,
                 probability: float,  # 变长交叉概率
                 gene_crossover_probability: float,  # 次级交叉概率
                 choose_variation_strategy: str,  # 变长策略
                 choose_gene_crossover: str,  # 次级交叉策略
                 para_bounds: dict):  # 参数上下界
        """
        :param probability: 变长交叉概率
            满足该概率，对染色体执行变长交叉，否则直接原样返回；

        :param gene_crossover_probability: 次级交叉概率
            满足该概率，对染色体上的等位基因执行次级交叉，否则不交叉；

        :param choose_variation_strategy: 变长策略
            - choose_variation_strategy = 'variation1': 变长1
            - choose_variation_strategy = 'variation2': 变长2
            - choose_variation_strategy = 'fixed': 定长
            - choose_variation_strategy = 'TBD': 待定(To be decided) -- 在execute()调用时决定

        :param choose_gene_crossover: 次级交叉策略
            - choose_gene_crossover = 'sbx': SBX交叉
            - choose_gene_crossover = 'swap': 互换交叉
            - choose_gene_crossover = 'nothing': 什么也不做交叉
            - choose_gene_crossover = 'TBD': 待定(To be decided) -- 在execute()调用时决定

        :param para_bounds: 参数的上下界信息（用于SBX交叉）

        """
        super(VariableLengthCrossover, self).__init__(probability=probability)

        self.choose_variation_strategy = choose_variation_strategy  # 变长策略
        self.choose_gene_crossover = choose_gene_crossover  # 基因位点交叉策略

        self.sbx_crossover = SBXCrossover(para_bounds, gene_crossover_probability, distribution_index=20.0)  # SBX交叉算子（待用）
        self.swap_crossover = SwapCrossover(probability=0.5)  # 交换交叉算子，概率设置为0.5不变（待用）
        self.do_nothing_crossover = DoNothingCrossover(probability=0.5)  # 什么也不做交叉，概率设置为0.5不变（待用）

    def execute(self, parents: List[MOEESolution], choose_variation_strategy: str = None, choose_gene_crossover: str = None) -> List[MOEESolution]:
        """
        - 此处的两个参数(choose_variation_strategy和choose_gene_crossover)默认为None，表示采用__init__()传入的相应参数来决定两个策略的选择；
        - 如果想通过在此处传入这两个参数来动态地决定两个策略的选择，那么__init__()传入的相应参数必须均为'TBD'，且此处必须按照两个策略的格式正确传入参数;

        举例：
            - 【进化过程中两个策略固定不变】：在交叉算子实例化时传入两个策略的参数值，并在execute中不传入任何参数；
            - 【进化过程中两个策略动态变化】（变长策略基于进化程度动态选择、次级交叉策略通过聚类间/聚类内动态选择）：在交叉算子实例化时两个策略的参数值均设置为'TBD'（待定），并在调用execute()执行变长交叉操作时，传入两个策略对应的参数值；

        """
        # 检查：交叉操作的对象必须是MOEESolution（即整个染色体/集成模型）
        Check.that(isinstance(parents[0], MOEESolution),
                   f'Solution invalid: {str(type(parents[0]))}, must be a MOEESolution !')
        Check.that(isinstance(parents[1], MOEESolution),
                   f'Solution invalid: {str(type(parents[0]))}, must be a MOEESolution !')

        # 确定两个策略的选择
        if self.choose_variation_strategy == 'TBD' and self.choose_gene_crossover == 'TBD':
            variation_strategy = choose_variation_strategy
            gene_crossover_strategy = choose_gene_crossover
        else:
            variation_strategy = self.choose_variation_strategy
            gene_crossover_strategy = self.choose_gene_crossover

        # 不满足变长交叉概率，直接返回parents
        if random.random() > self.probability:
            return parents
        else:
            """ 1.将长度相异的两个父代分为三段 """
            # 将mkSVR子学习机数目较少（染色体较短）的父代解深拷贝给解s1，数目较多（染色体较长）的深拷贝给解s2
            if parents[0].k < parents[1].k:
                offspring = [
                    copy.deepcopy(parents[0]), 
                    copy.deepcopy(parents[1])]
            else:
                offspring = [
                    copy.deepcopy(parents[1]), 
                    copy.deepcopy(parents[0])]

            s1 = offspring[0].variables  # 较短解
            s2 = offspring[1].variables  # 较长解

            # 将较长解拆成c21和c22两段（其中c21的长度和c1相等, c22是剩下的部分）
            idx = list(range(offspring[1].k))
            random.shuffle(idx)  # 随机打乱，使得与s1交叉的部分s2片段上的子学习机是随机抽取的
            s21 = s2.iloc[idx[:offspring[0].k]]
            s22 = s2.iloc[idx[offspring[0].k:]]

            """ 2. 对长度相同的两段染色体片段进行次级交叉 """
            # 定义次级交叉算子
            if gene_crossover_strategy == 'sbx':
                self.gene_crossover = self.sbx_crossover  # SBX参数交叉

            elif gene_crossover_strategy == 'swap':
                self.gene_crossover = self.swap_crossover  # 互换交叉

            elif gene_crossover_strategy == 'nothing':
                self.gene_crossover = self.do_nothing_crossover  # 什么也不做交叉

            else:
                raise AttributeError(
                    f'Input parameter choose_gene_crossover is invalid: {gene_crossover_strategy}.')

            # 将s1和s21（s2的前段）按照mkSVR块儿/基因位点对应匹配，并对每个匹配的等位基因（mkSVR子学习机对）按位点交叉
            for i in range(len(s1)):
                # 取两个行向量（两个mkSVR的参数向量）作为SBX交叉的父代
                p1, p2 = s1.iloc[i], s21.iloc[i]
                # 按位点交叉
                c = self.gene_crossover.execute([p1, p2])
                # 将新个体重新赋值给两个解
                s1.iloc[i], s21.iloc[i] = c[0], c[1]

            """ 3. 处理多余出来的一段染色体片段 """
            if variation_strategy == 'variation1':
                # 将三个染色体片段全部整合在一起，然后再随机分配给两个子代
                s = pd.concat([s1, s21, s22])  # 合并三段染色体
                idx = list(range(len(s)))
                random.shuffle(idx)  # 获取随机打乱的索引
                random_split_point = random.randint(1, len(s)-1)
                s1 = s.iloc[idx[:random_split_point]]
                s21 = s.iloc[idx[random_split_point:]]

            elif variation_strategy == 'variation2':
                # 将较长解c2剩下的一部分随机分配给两个新解s1'和s21'
                for i in range(len(s22)):
                    if random.random() <= 0.5:
                        # 这里是pd.DataFrame.append(pd.Series)
                        s1 = s1.append(s22.iloc[i])
                    else:
                        # 这里是pd.DataFrame.append(pd.Series)
                        s21 = s21.append(s22.iloc[i])

            elif variation_strategy == 'fixed':
                # 将s22放回原位（即仍拼接在s21后面）
                s21 = pd.concat([s21, s22])

            else:
                raise AttributeError(
                    f'Input parameter choose_variation_strategy is invalid: {variation_strategy}.')

            # 将交叉操作的结果赋值给子代solution的variables
            offspring[0].variables = s1
            offspring[1].variables = s21

            # 更新/维护solution内部变量的信息
            offspring[0].update_info()
            offspring[1].update_info()

            return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Variable-length crossover'


class SwapCrossover(Crossover):
    """ 
    互换交叉
        作用对象：基因位点/子学习机
    """

    def __init__(self, probability: float):
        super(SwapCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[S]) -> List[S]:
        rand = random.random()
        if rand <= self.probability:
            # 满足交叉概率，则进行交换
            offspring = [copy.deepcopy(parents[1]), copy.deepcopy(parents[0])]
            return offspring
        else:
            # 不满足交叉概率，则直接返回
            return copy.deepcopy(parents)

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Swap crossover'


class DoNothingCrossover(Crossover):
    """ 
    什么也不做交叉（直接返回原parents）
        作用对象：基因位点/子学习机
    """

    def __init__(self, probability: float):
        super(DoNothingCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[S]) -> List[S]:
        return copy.deepcopy(parents)

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Do nothing crossover'


class SBXCrossover(Crossover):
    """ 
    模拟二进制交叉(SBX) 
        作用对象：基因位点/子学习机

    """
    __EPS = 1.0e-14

    def __init__(self, para_bounds: dict, probability: float, distribution_index: float = 20.0):
        super(SBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index  # 公式中的n值，该值越大，交叉生成的子代与父代越接近
        if distribution_index < 0:
            raise Exception(
                "The distribution index is negative: " + str(distribution_index))
        self.para_bounds = para_bounds  # 参数的上下界

    def execute(self, parents: List[S]) -> List[S]:
        """
        :para parents: list([p1, p2]), 其中p1和p2是pd.Series. (不再传入两个solution对象)
        :para bounds: moo.solution.MOEESolution.para_bounds (字典类型).
        """
        Check.that(len(parents) == 2,
                   'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]),
                     copy.deepcopy(parents[1])]  # 初始化子代
        rand = random.random()

        # SBX算法涉及概率1（染色体重组概率）和概率2（条件重组概率），且强调概率1，把概率2设为定值0.5
        if rand <= self.probability:  # 概率1：染色体重组概率（满足该概率，执行交叉操作）

            for i in range(len(parents[0])):  # 逐个位点进行交叉操作
                value_x1, value_x2 = parents[0][i], parents[1][i]

                if random.random() <= 0.5:  # 概率2：条件重组概率（满足该概率，在染色体最小片段上执行交叉操作）

                    # 如果两个父代的差距大于阈值，我们才认为它们是不同的，此时再进行后续交叉
                    if abs(value_x1 - value_x2) > self.__EPS:
                        # 保证y1<y2
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound, upper_bound = self.para_bounds['low'][i], self.para_bounds['high'][i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - \
                            pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(
                                rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha),
                                        1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))  # 计算子代个体1

                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - \
                            pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow(
                                (rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha),
                                        1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))  # 计算子代个体2

                        # 修复超过边界的解
                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        # 将生成的两个个体随机分配给两个子代
                        if random.random() <= 0.5:
                            offspring[0][i] = c2
                            offspring[1][i] = c1
                        else:
                            offspring[0][i] = c1
                            offspring[1][i] = c2

                    # 如果两个父代的差距小于阈值，就认为它是相同的，此时不进行交叉，直接赋值给子代
                    else:
                        offspring[0][i] = value_x1
                        offspring[1][i] = value_x2

                # 不满足条件重组概率，不交叉
                else:
                    offspring[0][i] = value_x1
                    offspring[1][i] = value_x2

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'SBX crossover'


class FixedLengthCrossover(Crossover):
    """
    定长交叉算子
        作用对象：整个染色体/集成模型

    设计方案：
        把编码矩阵展开成一维向量，然后用普通的SBX交叉算子执行交叉操作
        
    """

    def __init__(self,
                 probability: float,  # 定长交叉概率
                 para_bounds: dict):  # 参数上下界
        super(FixedLengthCrossover, self).__init__(probability=probability)

        self.sbx_crossover = SBXCrossover(
            para_bounds, probability, distribution_index=20.0)  # SBX交叉算子

        self.swap_crossover = SwapCrossover(
            probability=0.5)  # 交换交叉算子，概率设置为0.5不变
    
    def execute(self, parents: List[MOEESolution], method: str = 'sbx') -> List[MOEESolution]:
        # 检查：交叉操作的对象必须是MOEESolution（即整个染色体/集成模型）
        Check.that(isinstance(parents[0], MOEESolution),
                   f'Solution invalid: {str(type(parents[0]))}, must be a MOEESolution !')
        Check.that(isinstance(parents[1], MOEESolution),
                   f'Solution invalid: {str(type(parents[0]))}, must be a MOEESolution !')

        # 不满足定长交叉概率，直接返回parents
        if random.random() > self.probability:
            return parents
        else:
            offspring = [
                copy.deepcopy(parents[0]), 
                copy.deepcopy(parents[1])]

            # 将参数矩阵(DataFrame)展开成一维向量(Series)
            s1 = offspring[0].variables.stack()
            s2 = offspring[1].variables.stack()

            if method == 'sbx':
                # 对一维向量执行SBX交叉
                c = self.sbx_crossover.execute([s1, s2])
                # 将交叉后的两个子代(Series)还原为参数矩阵(DataFrame)
                c1 = c[0].unstack()
                c2 = c[1].unstack()

            elif method == 'swap':
                # 对一维向量执行互换交叉（按位点互换）
                c1, c2 = copy.deepcopy(s1), copy.deepcopy(s2)
                for i in range(len(s1)):
                    c = self.swap_crossover.execute([s1.iloc[i], s2.iloc[i]])
                    c1.iloc[i], c2.iloc[i] = c[0], c[1]
                # 将交叉后的两个子代(Series)还原为参数矩阵(DataFrame)
                c1 = c1.unstack()
                c2 = c2.unstack()

            else:
                raise AttributeError(f'Input method is invalid: {method}')

            # 将交叉结果赋值给两个子代解
            offspring[0].variables = c1
            offspring[1].variables = c2

            # 更新/维护solution内部变量的信息
            offspring[0].update_info()
            offspring[1].update_info()

            return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Fixed-Length Crossover'

