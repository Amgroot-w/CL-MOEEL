import copy
import random

import pandas as pd

from core.operator import Mutation
from core.problem import MOEESolution
from core.util.checking import Check


class VariableLengthMutation(Mutation):
    """ 
    变长变异算子(MOEE) 
        作用对象：整个染色体/集成模型

    设计方案：
        - 变异策略1：增加一个基因位点/子学习机，染色体长度+1
        - 变异策略2：删除一个随机的基因位点/子学习机，染色体长度-1
        - 变异策略3：修改一个随机的基因位点/子学习机，染色体长度不变

    注意：
        - 对于增加、删除策略，其作用对象是基因位点/子学习机；
        - 对于修改策略，其作用对象是单个元素/子学习机的单个参数；
        传入的次级变异概率参数只在修改策略中发挥作用，它定义了每个元素被修改的概率；这个值应该大一点，因为对于增加和删除策略，
        一旦被1/3的概率选择到，是无条件执行的；而修改策略被1/3的概率选到后，对于矩阵（集成模型）中的每一个行向量（子学习机）的
        每一个元素值（子学习机参数），都需要先判断是否满足该次级变异概率，然后才决定是否对它做参数变异。

    """

    def __init__(self,
                 probability: float,  # 变长变异概率
                 gene_mutation_probability: float,  # 次级变异概率
                 para_bounds: dict  # 参数上下界
                 ):
        """
        :param probability: 变长变异概率
            满足此概率才执行变长变异操作
        :param gene_mutation_probability: 基因位点的变异概率
            次级变异算子的变异概率。即：在满足变长变异概率（一般为0.1）、且变异策略为“修改”（概率为1/3）的情况下，对矩阵中每个元素值进行变异的概率；
        :param para_bounds: 参数上下界信息
        """
        super(VariableLengthMutation, self).__init__(probability=probability)
        self.gene_mutator = PolynomialMutation(
            para_bounds=para_bounds, probability=gene_mutation_probability, distribution_index=0.2)

    def execute(self, parent: MOEESolution) -> MOEESolution:
        # 检查输入是否符合要求：必须是MOEESolution对象
        Check.that(isinstance(parent, MOEESolution),
                   f'Solution invalid: {type(parent)}, must be a MOEESolution !')

        if random.random() > self.probability:
            return parent  # 不满足变长变异概率，直接返回parent

        else:
            # child = copy.deepcopy(parent)  # 深拷贝parent给child
            rand = random.random()  # 生成突变随机数，从突变策略池中3选1
            if 0 <= rand < 1/3:
                """ 增加一个mkSVR子学习机 """
                parent = self.add(parent)
            elif 1/3 <= rand < 2/3:
                """ 删除一个mkSVR子学习机 """
                parent = self.delete(parent)
            else:
                """ 修改一个mkSVR子学习机 """
                parent = self.modify(parent)

            parent.update_info()  # 更新solution内部变量的信息
            return parent

    def add(self, solution: MOEESolution) -> MOEESolution:
        """ 变异策略1：增加一个基因位点（一个mkSVR子学习机），染色体长度+1 """
        new_mkSVR = pd.DataFrame(
            columns=solution.para_names,
            index=['mkSVR{}'.format(len(solution.variables))])

        new_mkSVR.iloc[0] = [random.uniform(
            solution.para_bounds['low'][i], solution.para_bounds['high'][i]) for i in range(solution.n)]

        solution.variables = solution.variables.append(new_mkSVR)
        return solution

    def delete(self, solution: MOEESolution) -> MOEESolution:
        """ 变异策略2：删除一个随机的基因位点（一个mkSVR子学习机），染色体长度-1 """
        if len(solution.variables) == 1:
            return solution  # 如果solution只有一个子学习机，那么不执行删除操作，直接原样返回
        else:
            delete_point = random.randint(
                0, len(solution.variables)-1)  # 随机选择删除节点
            # 在原始dataframe上直接drop
            solution.variables.drop(index=f'mkSVR{delete_point}', inplace=True)
            return solution

    def modify(self, solution: MOEESolution) -> MOEESolution:
        """ 变异策略3：修改一个随机的基因位点（一个mkSVR子学习机），染色体长度不变 """
        # for i in range(len(solution.variables)):
        #     # 此处调用次级变异算子，实现逐行（逐子学习机）修改操作
        #     solution.variables.iloc[i] = self.gene_mutator.execute(
        #         solution.variables.iloc[i])
        modify_point = random.randint(0, len(solution.variables)-1)  # 随机选择修改节点
        solution.variables.iloc[modify_point] = self.gene_mutator.execute(solution.variables.iloc[modify_point])        
        return solution

    def get_number_of_parents(self) -> int:
        return 1  # 变异操作对象是单个个体解

    def get_number_of_children(self) -> int:
        return 1  # 一次变异产生一个子代解

    def get_name(self):
        return 'Variable-length mutation'


class PolynomialMutation(Mutation):
    """ 
    多项式变异
        作用对象：基因位点/子学习机
    """

    def __init__(self, para_bounds: dict, probability: float, distribution_index: float = 0.20):
        super(PolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        self.para_bounds = para_bounds  # 参数的上下界

    def execute(self, parent: pd.Series) -> pd.Series:
        # 要求传入的parent类型为pd.Series
        Check.that(type(parent) is pd.Series,
                   "Solution type invalid: {}, must be pd.Series !".format(type(parent)))

        for i in range(len(parent)):
            rand = random.random()  # 对每一位都进行变异概率判断

            if rand <= self.probability:
                y = parent[i]
                yl, yu = self.para_bounds['low'][i], self.para_bounds['high'][i]

                if yl == yu:
                    y = yl  # 当前位置的上下界值相等，直接赋值为该值
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = random.random()
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * \
                            (pow(xy, self.distribution_index + 1.0))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * \
                            (pow(xy, self.distribution_index + 1.0))
                        deltaq = 1.0 - pow(val, mut_pow)

                    y += deltaq * (yu - yl)

                    # 修复超过边界的解
                    if y < self.para_bounds['low'][i]:
                        y = self.para_bounds['low'][i]
                    if y > self.para_bounds['high'][i]:
                        y = self.para_bounds['high'][i]

                parent[i] = y

        return parent

    def get_number_of_parents(self) -> int:
        return 1  # 变异操作对象是单个个体解

    def get_number_of_children(self) -> int:
        return 1  # 一次变异产生一个子代解

    def get_name(self):
        return 'Polynomial mutation'


class FixedLengthMutation(Mutation):
    """ 
    定长变异算子
        作用对象：整个染色体/集成模型

    设计方案：
        - 依次对编码矩阵的每一行（每一个自学习机）执行多项式变异
    """

    def __init__(self,
                 probability: float,  # 定长变异概率
                 para_bounds: dict  # 参数上下界
                 ):
        super(FixedLengthMutation, self).__init__(probability=probability)
        self.gene_mutator = PolynomialMutation(
            para_bounds=para_bounds, probability=probability, distribution_index=0.2)

    def execute(self, parent: MOEESolution) -> MOEESolution:
        # 检查输入是否符合要求：必须是MOEESolution对象
        Check.that(isinstance(parent, MOEESolution),
                   f'Solution invalid: {type(parent)}, must be a MOEESolution !')
        # child = copy.deepcopy(parent)  # 深拷贝parent给child
        s = parent.variables.stack()  # 将参数矩阵展开为一维向量
        c = self.gene_mutator.execute(s)  # 对长向量执行多项式变异
        parent.variables = c.unstack()  # 将一维向量恢复为参数矩阵

        parent.update_info()  # 更新solution内部变量的信息
        return parent

    def get_number_of_parents(self) -> int:
        return 1  # 变异操作对象是单个个体解

    def get_number_of_children(self) -> int:
        return 1  # 一次变异产生一个子代解

    def get_name(self):
        return 'Fixed-length mutation'
