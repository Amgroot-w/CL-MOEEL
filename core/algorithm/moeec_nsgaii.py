import random
from itertools import combinations
from typing import List
import numpy as np
import pandas as pd
from core.operator import Selection, VariableLengthCrossover, VariableLengthMutation
from core.problem import MOEEProblem, MOEESolution
from core.util.initialization import Initializer
from core.util.replacement import ObjectiveBasedSlackBinaryTournamentReplacement, \
    RankingAndDensityEstimatorReplacement, Replacement
from core.util.strategy import CrossoverStrategyDecider, DecideVariationStrategyByEpoch, \
    DecideVariationStrategyByRandom, DecideVariationStrategyBySelfAdaptive
from .moee_algorithm import MOEEC_Algorithm


class MOEEC_NSGAII_v1(MOEEC_Algorithm):
    """
    MOEEC-NSGAII算法 (v1版本)

    Feature:
        - 在MOEE_NSGAII算法基础上，引入了聚类；
    
    算法细节：
        Step 1. 选择：用基于RMSE的松弛二元锦标赛选择算子，对三个子种群依次执行选择过程；

        Step 2. 交叉：先执行类内交叉，用和NSGAII相同的方式，产生一对对的parents，并对每一对parent做变长交叉得到children（变长交叉算子的变长策略由变长策略决策器产生，次级交叉策略选择SBX参数交叉）；类内交叉完成后，用得到的具有相同子代解个数的三个子种群，来做类间交叉，由于有3个子种群，因此共有C32=3种组合方式，在每种组合方式中，两个子种群先进行解的匹配，即以解数目较少的子种群的大小为准，在另一个数目较多的子种群中取出相同数量的个体，构成parents对，然后做类间交叉（变长交叉算子的变长策略由变长策略决策器产生，次级交叉策略选择Swap互换交叉），并用得到的子代个体替换掉父代个体，从而完成当前组合情况下的两个子种群的解的更新，以此类推，其他两个组合情况也按照这个步骤执行。所有的类间交叉都完成后，得到3个子种群（它们各自都经历了1次类内交叉、2次类间交叉）；
        
        Step 3. 变异：提取上一步得到的3个子种群中的全部个体解，组合成一个完整的大的种群，并依次对这个大种群中的所有解进行变长变异；
        
        Step 4. 修复解：对经历完上述所有步骤的个体解，依次执行修复操作，全部修复完成后，即得到最终的子代种群。
    
    配置：
        - selection: 用相同的选择算子从每个类别中选择出相应数量的个体解
        - reproduction: 类内交叉&类间交叉
        - replacement: 基于非支配排序和拥挤距离的环境选择（NSGA-II）
    """

    def __init__(self,
                 problem: MOEEProblem,  # 问题
                 selection: Selection,  # 选择算子
                 crossover: VariableLengthCrossover,  # 变长交叉算子
                 mutation:  VariableLengthMutation,   # 变长变异算子
                 population_size: int,  # 种群大小
                 max_epoch: int,  # 进化代数
                 initializer: Initializer):  # 种群初始化器
        super(MOEEC_NSGAII_v1, self).__init__(
            problem=problem,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            population_size=population_size,
            offspring_population_size=population_size,  # 该算法中子种群大小等于种群大小
            max_epoch=max_epoch,
            initializer=initializer)

    def selection(self, population: List[MOEESolution]) -> List[MOEESolution]:
        """
        基于聚类的：选择
            用相同的选择算子从每个类别中选择出相应数量的个体解
        """
        # 首先整理得到各个聚类的子种群
        sub_populations = {}
        for s in population:
            try:
                sub_populations[s.attributes['cluster_label']].append(s)
            except KeyError:
                sub_populations[s.attributes['cluster_label']] = [s]

        # 从每个聚类种群中选择出各自的交配种群
        mating_population = []  # 初始化父代种群
        for key in sub_populations.keys():
            sub_population = sub_populations[key]  # 获取子聚类种群

            # 从该子聚类种群中选择出具有相同数目的个体并入父代种群中
            for _ in range(len(sub_population)):
                solution = self.selection_operator.execute(sub_population)  # 选择算子
                mating_population.append(solution)

        return mating_population

    def reproduction(self, mating_population: List[MOEESolution]) -> List[MOEESolution]:
        """
        基于聚类的: 交叉 & 变异
            步骤：聚类内部交叉 -> 聚类之间交叉 -> 变异 -> 修复解
        """
        # 整理得到从各个聚类中选择出来的父代种群
        sub_mating_populations = {}
        for s in mating_population:
            try:
                sub_mating_populations[s.attributes['cluster_label']].append(s)
            except KeyError:
                sub_mating_populations[s.attributes['cluster_label']] = [s]

        # 1. 交叉
        # ** 变长策略: 随机确定
        VariationStrategyDecider = DecideVariationStrategyByRandom()  
        variation_strategy = VariationStrategyDecider.execute()
        # ** 变长策略：基于Epoch确定
        # VariationStrategyDecider = DecideVariationStrategyByEpoch(max_epoch=self.max_epoch)
        # variation_trategy = VariationStrategyDecider.execute(epoch=self.epoch)

        # 1.1 聚类内部交叉: 带有“SBX参数交叉”的变长交叉算子
        sub_offspring_populations = {}  # 初始化子代种群  
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()  # 交叉算子所需的父代个体数     
        for key in sub_mating_populations.keys():
            sub_mating_population = sub_mating_populations[key]  # 获取子聚类种群
            sub_offspring_population = []  # 初始化子代种群
                
            # 检查：若种群个体数不能被number_of_parents_to_combine整除，则去掉最后几个（并将它们直接并入offspring中），保证其能被整除
            n = len(sub_mating_population) % number_of_parents_to_combine  # 计算余数
            number = len(sub_mating_population) - n  # 类内交叉实际用到的个体数
            if  n != 0:
                sub_offspring_population.extend(sub_mating_population[-n:])  # 将多出来的n个没用到的个体直接并入offspring中
            
            # 产生parents对，并做变长交叉
            for i in range(0, number, number_of_parents_to_combine):
                parents = []
                for j in range(number_of_parents_to_combine):
                    # 第 i ~ (i + number_of_parents_to_combine) 个父代个体，作为第i对parents
                    parents.append(sub_mating_population[i + j])

                # 变长交叉: (1)变长交叉策略由决策器动态产生; (2)次级交叉策略选择SBX参数交叉
                offspring = self.crossover_operator.execute(parents=parents, 
                                                            choose_variation_strategy=variation_strategy,
                                                            choose_gene_crossover='sbx')
                sub_offspring_population.extend(offspring)  # 产生的新个体并入子代种群中
            
            sub_offspring_populations[key] = sub_offspring_population  # 记录当前聚类子种群得到的offspring种群

        # 1.2 聚类之间交叉: 带有“互换交叉”的变长交叉算子
        # 注意：聚类之间交叉不生成新的变量，而是直接在原始变量上操作！
        cluster_combinations = combinations(sub_offspring_populations.keys(), 2)  # 产生所有聚类子种群之间的组合情况（C32=3种）
        # 对每种组合情况做类间交叉
        for key1, key2 in cluster_combinations:
            sub_mating_population1 = sub_offspring_populations[key1]  # 子聚类种群1
            sub_mating_population2 = sub_offspring_populations[key2]  # 子聚类种群2

            random.shuffle(sub_mating_population1)  # 随机打乱子聚类种群1
            random.shuffle(sub_mating_population2)  # 随机打乱子聚类种群2

            n1, n2 = len(sub_mating_population1), len(sub_mating_population2)  # 各自的种群大小

            for i in range(min(n1, n2)):
                # 从两个子聚类种群中各取一个父代个体
                parents = [sub_mating_population1[i], sub_mating_population2[i]]
                # 变长交叉: (1)变长交叉策略由决策器动态产生; (2)次级交叉策略选择互换交叉
                offspring = self.crossover_operator.execute(parents=parents, 
                                                            choose_variation_strategy=variation_strategy,
                                                            choose_gene_crossover='swap')
                # 用生成的子代替换两个父代个体
                sub_mating_population1[i], sub_mating_population2[i] = offspring

        # 2. 变异
        offspring_population = []  # 初始化最终的子代种群
        # 取出经过交叉操作之后的各个聚类子种群
        for population in sub_offspring_populations.values():
            offspring_population.extend(population)
        # 对所有个体执行变异操作
        for solution in offspring_population:
            solution = self.mutation_operator.execute(solution)  # 变异
            solution = self.problem.fix_solution(solution)  # 修复解

        return offspring_population

    def replacement(self, population: List[MOEESolution], offspring_population: List[MOEESolution]) -> List[List[MOEESolution]]:
        """ 环境选择 """
        # 基于非支配排序和拥挤距离的环境选择（NSGA-II）
        enviroment_selector = RankingAndDensityEstimatorReplacement()
        return enviroment_selector.replace(population, offspring_population)

    def get_name(self) -> str:
        return 'MOEEC_NSGAII_v1'


class MOEEC_NSGAII_v3(MOEEC_NSGAII_v1):
    """
    MOEEC-NSGAII-v3算法

    Feature:
        - 基于知识引导的选择(Knowledge-guided selection)：对不同的聚类子种群，基于其“秩平均”计算得到其“优势”目标函数
    
    配置：
        - selection（重写）: 对不同的聚类子种群采用不同的选择策略（基于子种群的优势）
        - reproduction（继承自v1）: 类内交叉&类间交叉
        - replacement（重写）: 基于RMSE和松弛二元锦标赛选择的环境选择（EvoCNN）
    """

    def __init__(self,
                problem: MOEEProblem,  # 问题
                selection: Selection,  # 选择算子
                crossover: VariableLengthCrossover,  # 变长交叉算子
                mutation:  VariableLengthMutation,   # 变长变异算子
                population_size: int,  # 种群大小
                max_epoch: int,  # 进化代数
                initializer: Initializer):  # 种群初始化器
        super(MOEEC_NSGAII_v3, self).__init__(
            problem=problem,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            population_size=population_size,
            max_epoch=max_epoch,
            initializer=initializer)
    
    def selection(self, population: List[MOEESolution]) -> List[MOEESolution]:
        """ 重写：选择 """
        # 首先整理得到各个聚类的子种群
        sub_populations = {}
        for s in population:
            try:
                sub_populations[s.attributes['cluster_label']].append(s)
            except KeyError:
                sub_populations[s.attributes['cluster_label']] = [s]

        # 获取当前种群的“优势”目标函数，并基于该函数进行后续的选择操作
        advantage_objs = self.get_advantage_objs(population)
        self.recorder.record_cluster(info=advantage_objs, epoch=self.epoch+1)  # 记录该信息

        # 从每个聚类种群中选择出各自的交配种群
        mating_population = []  # 初始化父代种群
        for key in sub_populations.keys():
            sub_population = sub_populations[key]  # 获取聚类子种群
            advantage_obj = advantage_objs[key]    # 获取聚类子种群的优势目标函数

            # 从该子聚类种群中选择出具有相同数目的个体并入父代种群中
            for _ in range(len(sub_population)):
                solution = self.selection_operator.execute(sub_population, prefer_obj=advantage_obj)  # 选择算子
                mating_population.append(solution)

        return mating_population       

    def get_advantage_objs(self, population: dict) -> dict:
        """
        基于“秩平均”来计算聚类子种群的“优势”目标函数

            - 输入: 完整的种群（MOEESolution列表）
            - 输出: 各聚类子种群的优势目标函数（字典变量）
        """
        objs = pd.DataFrame([s.objectives for s in population], columns=['RMSE', 'Diversity', 'Complexity'])
        ranks = pd.DataFrame(columns=['RMSE', 'Diversity', 'Complexity'])

        for obj_name in objs.columns:
            ranks[obj_name] = self.get_rank(objs[obj_name])
        
        cluster_mean_ranks = {
            0: pd.DataFrame([[0, 0, 0]], columns=['RMSE', 'Diversity', 'Complexity']),
            1: pd.DataFrame([[0, 0, 0]], columns=['RMSE', 'Diversity', 'Complexity']),
            2: pd.DataFrame([[0, 0, 0]], columns=['RMSE', 'Diversity', 'Complexity'])
        }  # 初始化

        for i, s in enumerate(population):
            cluster_mean_ranks[s.attributes['cluster_label']] += ranks.iloc[i, :] / len(population)
        
        # 整理得到3x3矩阵，然后按照优先级整理出各个种群的优势目标函数
        temp = np.array([values.values[0] for values in cluster_mean_ranks.values()])
        rank_obj_matrix = pd.DataFrame(columns=['RMSE', 'Diversity', 'Complexity'])
        for i, col in enumerate(rank_obj_matrix.columns):
            rank_obj_matrix[col] = self.get_rank(temp[:, i])

        advantage_objs = {}
        for i in range(3):
            df = rank_obj_matrix.iloc[i, :].sort_values()

            if df[0] == df[1]:
                if df[1] == df[2]:
                    obj = 'RMSE'  # 三个目标函数都是最佳，则优先选择RMSE
                else:
                    if df.index[2] == 'RMSE':
                        obj = 'Complexity'  # 若Diversity和Complexity都是最佳，则优先选择Complexity
                    else:
                        obj = 'RMSE'  # 若两个都是最佳的目标函数中有一个是RMSE，则优先选择RMSE
            else:
                obj = df.index[0]  # 如果三个目标函数的排名各不相同，则正常返回排第一个的目标函数
            
            advantage_objs[i] = obj
        
        return advantage_objs

    @staticmethod
    def get_rank(vector):
        rank = np.zeros(len(vector))
        sorted_index = np.argsort(vector)
        for i, index in enumerate(sorted_index):
            rank[index] = i
        return rank

    def replacement(self, population: List[MOEESolution], offspring_population: List[MOEESolution]) -> List[List[MOEESolution]]:
        """ 重写：环境选择 """
        # 基于非支配排序和拥挤距离的环境选择（NSGA-II）
        enviroment_selector = RankingAndDensityEstimatorReplacement()

        return enviroment_selector.replace(population, offspring_population)

    def get_name(self) -> str:
        return 'MOEEC_NSGAII_v3'


class MOEEC_NSGAII_v5(MOEEC_NSGAII_v3):
    """
    MOEEC-NSGAII-v5算法

    Feature:
        - 基于知识引导的选择 - KnowledgeBasedSelection
        - 变长交叉 - VariableLengthCrossover
        - 变长变异- VariableLengthMutation

    配置：
        - init_progress（重写）：配置新的自适应策略决策器初值；
        - selection（重写）: 可以选择：1.基于知识引导的选择（调用v3）; 2.其他传统的选择算子（调用v1）；
        - reproduction（重写）: 包含了类内交叉（新）和类间交叉（新），
            可以选择：1.只用类内；2.只用类间；3.先类内后类间；4.随机确定（默认）5.基于成功率自适应确定；
        - replacement（重写）: NSGA-II的环境选择；
    """

    def __init__(self,
                 problem: MOEEProblem,  # 问题
                 selection: Selection,  # 选择算子
                 crossover: VariableLengthCrossover,  # 变长交叉算子
                 mutation:  VariableLengthMutation,   # 变长变异算子
                 population_size: int,  # 种群大小
                 max_epoch: int,  # 进化代数
                 crossover_method: str,  # 交叉方法
                 learning_period: int, # 学习周期
                 initializer: Initializer  # 种群初始化器
                 ):
        super(MOEEC_NSGAII_v5, self).__init__(
            problem=problem,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            population_size=population_size,
            max_epoch=max_epoch,
            initializer=initializer)
        # 交叉方法
        self.crossover_method = crossover_method
        # 自适应交叉策略决策器
        self.crossover_strategy_decider = CrossoverStrategyDecider(learning_period=learning_period)
        
    def init_progress(self):
        """ 初始化进化过程 """
        # 初始化策略及其flag
        for solution in self.pop:
            # 自适应确定交叉策略时，需要记录其策略信息
            solution.attributes['strategy'] = None  # 策略初始化为None
            # 自适应选择交叉策略时，需要用到的一个flag变量，该变量为0表示是（未经过变长交叉的）父代种群，该变量为1表示是（经过变长交叉的）子代种群
            solution.attributes['strategy_flag'] = 0  # flag初始化为0

    def selection(self, population: List[MOEESolution]) -> List[MOEESolution]:
        """ 重写：选择 """
        if 'Slack' in self.selection_operator.get_name():
            # 基于本次进化种群的目标函数值，来更新三个目标函数的阈值
            objs = pd.DataFrame([s.objectives for s in self.pop], columns=['RMSE', 'Diversity', 'Complexity'])
            thresholds = (objs.max() - objs.min()) * 0.005
            # 设置该选择算子的松弛比较器的三个目标函数的阈值
            self.selection_operator.comparator.update_threshold(thresholds['RMSE'], thresholds['Diversity'], thresholds['Complexity'])
            # 记录三个目标函数阈值
            self.recorder.record_log_thresholds(thresholds=list(thresholds), epoch=self.epoch)
            # 基于知识引导的选择算子：从子种群中依次地基于偏好目标函数进行选择
            return MOEEC_NSGAII_v3.selection(self, population)
        else:
            # 传统选择算子：从聚类子种群中分别执行传统选择算子，然后合并
            return MOEEC_NSGAII_v1.selection(self, population)

    def reproduction(self, mating_population: List[MOEESolution]) -> List[MOEESolution]:
        """ 重写：交叉&变异 """

        def __InsideClusterCrossover(sub_mating_populations):
            """
            聚类内部交叉
                --- 参数交叉、子学习机位置不变
                变长策略：'fixed' (定长)
                次级交叉算子：'sbx' (参数交叉)
            """
            sub_offspring_populations = {}  # 初始化子代种群  
            number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()  # 交叉算子所需的父代个体数   
            for key in sub_mating_populations.keys():
                sub_mating_population = sub_mating_populations[key]  # 获取子聚类种群
                sub_offspring_population = []  # 初始化子代种群
                    
                # 检查：若种群个体数不能被number_of_parents_to_combine整除，则去掉最后几个（并将它们直接并入offspring中），保证其能被整除
                n = len(sub_mating_population) % number_of_parents_to_combine  # 计算余数
                number = len(sub_mating_population) - n  # 类内交叉实际用到的个体数
                if  n != 0:
                    sub_offspring_population.extend(sub_mating_population[-n:])  # 将多出来的n个没用到的个体直接并入offspring中
                
                # 产生parents对，并做变长交叉
                for i in range(0, number, number_of_parents_to_combine):
                    parents = []
                    for j in range(number_of_parents_to_combine):
                        # 第 i ~ (i + number_of_parents_to_combine) 个父代个体，作为第i对parents
                        parents.append(sub_mating_population[i + j])

                    # 类内变长交叉: (1)变长交叉策略; (2)次级交叉策略
                    offspring = self.crossover_operator.execute(parents=parents,
                                                                choose_variation_strategy='fixed',
                                                                choose_gene_crossover='sbx')
                    # 将策略信息记录在解的属性中
                    for solution in offspring:
                        solution.attributes['strategy'] = 'inside'  # 记录生成该子代解的交叉策略--类内交叉
                        solution.attributes['strategy_flag'] = 1  # 更新改个体解的flag，表示此时改解已从父代转变为子代

                    sub_offspring_population.extend(offspring)  # 产生的新个体并入子代种群中
                
                sub_offspring_populations[key] = sub_offspring_population  # 记录当前聚类子种群得到的offspring种群
            
            return sub_offspring_populations
        
        def __OutsideClusterCrossover(sub_matting_populations):
            """
            聚类之间交叉
                --- 参数不变、子学习机全部随机分配
                变长策略：'variation1' (定长1)
                次级交叉算子：'nothing'(什么也不做)
            """
            # 将3个子种群各自均分为两半，一共得到6个小种群
            i = 1
            pops = {}  # 初始化，用来存储6个小种群
            for key in sub_matting_populations.keys():
                random.shuffle(sub_matting_populations[key])  # 先随机打乱该子种群
                num = len(sub_matting_populations[key]) // 2
                pops[i] = sub_matting_populations[key][:num]  # 拆分得到前面一半种群
                pops[i+1] = sub_matting_populations[key][num:]  # 拆分得到后面一半种群
                i += 2

            # 6个小种群，进行3种类间组合交叉
            for key1, key2 in [[1, 5], [2, 3], [4, 6]]:
                for i in range(min(len(pops[key1]), len(pops[key2]))):
                    # 从两个子聚类种群中各取一个父代个体
                    parents = [pops[key1][i], pops[key2][i]]
                    # 类间变长交叉: (1)变长交叉策略; (2)次级交叉策略
                    offspring = self.crossover_operator.execute(parents=parents, 
                                                                choose_variation_strategy='variation1',
                                                                choose_gene_crossover='nothing')
                    # 将策略信息记录在解的属性中
                    for solution in offspring:
                        solution.attributes['strategy'] = 'outside'  # 记录生成该子代解的交叉策略--类间交叉
                        solution.attributes['strategy_flag'] = 1  # 更新改个体解的flag，表示此时改解已从父代转变为子代

                    # 用生成的子代替换两个父代个体
                    parents[0].replace_by(offspring[0])
                    parents[1].replace_by(offspring[1])

            return sub_matting_populations
        
        def __VariableLengthMutation(sub_offspring_populations):
            """
            变长变异
            """
            offspring_population = []  # 初始化最终的子代种群
            # 取出经过交叉操作之后的各个聚类子种群
            for population in sub_offspring_populations.values():
                offspring_population.extend(population)
            # 对所有个体执行变异操作
            for solution in offspring_population:
                solution = self.mutation_operator.execute(solution)  # 变异
                solution = self.problem.fix_solution(solution)  # 修复解

            return offspring_population
        
        """ 交叉 """
        sub_mating_populations = {}
        for s in mating_population:
            try:
                sub_mating_populations[s.attributes['cluster_label']].append(s)
            except KeyError:
                sub_mating_populations[s.attributes['cluster_label']] = [s]

        # 【方案1】先做类内交叉 --> 再做类间交叉（原v4版本算法的思路）
        if self.crossover_method == 'both':
            sub_offspring_populations = __InsideClusterCrossover(sub_mating_populations)
            sub_offspring_populations = __OutsideClusterCrossover(sub_offspring_populations)
        
        # 【方案2】只做类内交叉
        elif self.crossover_method == 'inside':
            sub_offspring_populations = __InsideClusterCrossover(sub_mating_populations)

        # 【方案3】只做类间交叉
        elif self.crossover_method == 'outside':
            sub_offspring_populations = __OutsideClusterCrossover(sub_mating_populations)

        # 【方案4】随机选择：类内交叉 or 类间交叉
        elif self.crossover_method == 'random':
            if random.random() < 0.5:
                sub_offspring_populations = __InsideClusterCrossover(sub_mating_populations)
            else:
                sub_offspring_populations = __OutsideClusterCrossover(sub_mating_populations)
        
        # 【方案5】自适应选择：类内交叉 or 类间交叉
        elif self.crossover_method == 'adaptive':
            strategy = self.crossover_strategy_decider.execute()  # 自适应确定本次迭代的交叉策略：类内(inside) or 类间(outside)
            if strategy == 'inside':
                sub_offspring_populations = __InsideClusterCrossover(sub_mating_populations)
            elif strategy == 'outside':
                sub_offspring_populations = __OutsideClusterCrossover(sub_mating_populations)

        else:
            raise AttributeError(f'Input parameter error: {self.crossover_method}')
        
        """ 变异 """
        offspring_population = __VariableLengthMutation(sub_offspring_populations)
        
        return offspring_population

    def replacement(self, population: List[MOEESolution], offspring_population: List[MOEESolution]) -> List[MOEESolution]:
        """ 重写：环境选择 """
        # 先将父代种群中存储的策略信息清空，保证next_pop中的所有关于策略的信息均来自新生成的子代种群offspring_population
        for solution in population:
            solution.attributes['strategy'] = None  # 记录生成该子代解的变长策略
            solution.attributes['strategy_flag'] = 0  # 更新改个体解的flag，表示此时改解已从父代转变为子代
        
        # 自然选择得到下一代种群
        enviroment_selector = RankingAndDensityEstimatorReplacement()  # 基于非支配排序和拥挤距离的环境选择（NSGA-II）
        next_pop = enviroment_selector.replace(population, offspring_population)

        # 根据种群1（生成的子代种群）和种群2（自然选择后进入下一代的种群）来更新策略决策器的两个memory
        self.crossover_strategy_decider.update_memory(pop1=offspring_population, pop2=next_pop)

        # 记录每一次迭代更新过后，策略决策器中各个策略的选择概率
        p = [self.crossover_strategy_decider.p1, self.crossover_strategy_decider.p2]
        self.recorder.record_log_strategy_p(p=p, epoch=self.epoch+1)

        return next_pop

    def get_log(self) -> dict:
        """ 获取记录的算法信息 """
        log = {
            # 基本信息
            'algorithm': self.get_name(),  # 算法名
            'problem': self.problem.get_name(),  # 问题名
            'dataset': self.problem.dataset_name,  # 数据集名
            # 算法配置
            'meta_model': self.problem.meta_model_selection,  # meta模型选择
            'div_method': self.problem.div_method,  # 分散性指标计算方法
            'crossover_method': self.crossover_method,  # 交叉方法
            'selection_operator': self.selection_operator.get_name(), # 选择算子名称
            'crossover_operator': self.crossover_operator.get_name(), # 交叉算子名称
            'mutation_operator': self.mutation_operator.get_name(), # 变异算子名称
            # 超参数
            'population_size': self.population_size,  # 种群大小
            'max_epoch': self.max_epoch,  # 最大迭代次数
            'learning_period': self.crossover_strategy_decider.LP,  # 学习周期
            # 运行结果
            'runtime': self.runtime,  # 运行时间
            'pop': self.recorder.get_log_pop(),  # 所有进化代数的种群分布
            'cluster_info': self.recorder.get_log_cluster(),  # 每个聚类子种群的优势目标函数
            'strategy_p': self.recorder.get_log_strategy_p(),  # 进化过程中三个变长策略的选择概率的变化情况
            'log_thresholds': self.recorder.get_log_thresholds()  # 获取三个目标函数的变化情况
        }

        return log

    def get_name(self) -> str:
        return 'MOEEC_NSGAII_v5'

