import os
import pickle
import time
import yaml
import argparse
import sys; sys.path.append(os.getcwd())
import warnings; warnings.filterwarnings("ignore")

from core.problem import MOEEProblem
from core.algorithm import MOEEC_NSGAII_v5
from core.operator import RouletteWheelSelection, BinaryTournamentSelection, BestSolutionSelection, \
    RandomSolutionSelection, SlackBinaryTournamentSelection, SlackBinaryTournamentSelection_v1, \
    SlackBinaryTournamentSelection_v2, VariableLengthCrossover, VariableLengthMutation
from core.util.initialization import RandomInitializer
from core.dataset import get_dataset

"""
MOEEC-NSGAII-v5算法测试流程
    - 选择算子：基于知识引导的选择
    - 交叉算子：变长交叉
    - 变异算子：变长变异
    - 环境选择：基于支配关系和拥挤距离的环境选择（NSGA-II）
"""

# 读取配置文件
params = yaml.load(open('config/params.yml', encoding='utf8'), Loader=yaml.FullLoader)

parser = argparse.ArgumentParser('MOEEC_NSGAII_v5 Experiments')
parser.add_argument('--dataset', type=str, default='UCI14_ResidentialBuilding',
                    help="Choose dataset: UCI14_ResidentialBuilding or Si_predict")
parser.add_argument('--pop', type=int, default=params['pop_size'], 
                    help="Population size")
parser.add_argument('--epoch', type=int, default=params['max_epoch'],
                    help="Maximum evolutionary epoch")
parser.add_argument('--lp', type=int, default=params['learning_period'],
                    help="Learning period in Self-adaptive strategy")
parser.add_argument('--exp_nums', type=int, default=params['exp_nums'],
                    help="Total number of experiments")
parser.add_argument('--meta_model', type=str, default=params['meta_model'],
                    help="The meta model that ensemble the base learners' output: WeightSum, Linear, MLP, SVR")
parser.add_argument('--div_method', type=str, default=params['div_method'],
                    help="The Diversity definition of the solution in population")
parser.add_argument('--selection', type=str, default=params['selection'],
                    help="The selection method"),
parser.add_argument('--crossover', type=str, default=params['crossover'],
                    help="The crossover method"),
parser.add_argument('--info', type=str, default='',
                    help="Specific experiment information that will be added in the result filename")
parser.add_argument('--label', type=str, default='',
                    help="Specific experiment label that will be treated as an independent dir in the result path")
args = parser.parse_args()  # 解析参数

# 选择算子
if args.selection == 'Roulette':
    selection = RouletteWheelSelection()  # 轮盘赌选择
elif args.selection == 'Best':
    selection = BestSolutionSelection()  # （基于支配关系的）最优解选择
elif args.selection == 'Random':
    selection = RandomSolutionSelection()  # 随机选择
elif args.selection == 'BTS':
    selection = BinaryTournamentSelection()  # （基于支配关系的）二元锦标赛选择
elif args.selection == 'SlackBTS':
    selection = SlackBinaryTournamentSelection()  # （基于知识引导的）松弛二元锦标赛选择
elif args.selection == 'SlackBTS_v1':
    selection = SlackBinaryTournamentSelection_v1()  # （基于知识引导的）松弛二元锦标赛选择-改进1
elif args.selection == 'SlackBTS_v2':
    selection = SlackBinaryTournamentSelection_v2()  # （基于知识引导的）松弛二元锦标赛选择-改进2
else:
    raise AttributeError('Input error: selection')

# 算法名称
algorithm_name = 'MOEEC_NSGAII_v5'   
# 时间戳-年/月/日
time_stamp = time.strftime('%Y-%m-%d', time.localtime(time.time()))
# 额外信息
info = f'_{args.info}' if args.info != '' else args.info
# 测试的标签
label = f'{args.label}/' if args.label != '' else args.label
# 保存重复试验结果的根目录
save_path = f'models/{args.dataset}/{label}{algorithm_name}{info}_{time_stamp}_kfold'

# 算法运行Pipeline
def run_moeec_nsgaii_v5(problem, save_path, running_info, fold):
    """ 
    MOEEC-NSGAII-v5算法的执行过程
    """
    # 交叉参数
    crossover_probability = 1.0  # 变长交叉概率
    gene_crossover_probability = 1.0  # 次级交叉概率
    choose_variation_strategy = 'TBD'  # 变长交叉策略
    choose_gene_crossover = 'TBD'  # 次级交叉策略

    # 变异参数
    mutate_probability = 0.1  # 变长变异概率
    gene_mutation_probability = 0.5  # 次级变异概率
    
    # 进化参数
    population_size = args.pop  # 种群大小
    max_epoch = args.epoch  # 进化代数
    learning_period = args.lp  # 学习周期（自适应变长策略决策器）
    crossover_method = args.crossover  # 交叉方法

    # 种群初始化器：随机初始化
    initializer = RandomInitializer(population_size=population_size, problem=problem)

    algorithm = MOEEC_NSGAII_v5(
        problem=problem,  # 问题
        selection=selection,  # 选择算子
        crossover=VariableLengthCrossover(crossover_probability,
                                          gene_crossover_probability,
                                          choose_variation_strategy,
                                          choose_gene_crossover,
                                          problem.para_bounds),  # 变长交叉
        mutation=VariableLengthMutation(mutate_probability,
                                        gene_mutation_probability,
                                        problem.para_bounds),  # 变长变异
        population_size=population_size,  # 种群大小
        max_epoch=max_epoch,  # 进化代数
        crossover_method=crossover_method,  # 交叉方法
        learning_period=learning_period,  # 学习周期
        initializer=initializer  # 种群初始化器
    )

    algorithm.run(running_info)  # 运行

    results = algorithm.get_result()  # 获取进化结果
    log = algorithm.get_log()  # 获取进化日志

    # 保存路径
    save_dir = '{}/fold{}_{}_Runtime_{:.2}min'.format(
        save_path,  # 总路径
        fold,  # 第j折交叉验证
        time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())),  # 时间戳
        log['runtime']/60  # 进化用时
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 若路径不存在则创建路径

    with open(save_dir + '/results.pkl', 'wb') as f:
        pickle.dump(results, f)  # 保存进化结果

    with open(save_dir + '/log.pkl', 'wb') as f:
        pickle.dump(log, f)  # 保存进化日志


if __name__ == '__main__':
    start = time.time()
    for i in range(args.exp_nums):
        exp_time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))  # 时间戳
        for j in range(params['kfold_n_splits']):
            if args.info == '':
                running_info = "Dataset:{}, Model:{}, Time:{}, Exp:{}/{}, Fold:{}/{}:".format(
                    args.dataset, algorithm_name, 
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 
                    i+1, args.exp_nums, 
                    j+1, params['kfold_n_splits']
                )
            else:    
                running_info = "Dataset:{}, Model:{}, Info:{}, Time:{}, Exp:{}/{}, Fold:{}/{}:".format(
                    args.dataset, algorithm_name, args.info, 
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 
                    i+1, args.exp_nums, 
                    j+1, params['kfold_n_splits']
                )
            # 获取数据
            train_x, val_x, _, train_y, val_y, _ = get_dataset(args.dataset).get_split_data(method='kfold', fold=j)
            # 定义问题
            problem = MOEEProblem(dataset_name=args.dataset,
                                  train_x=train_x,
                                  val_x=val_x,
                                  train_y=train_y,
                                  val_y=val_y,
                                  meta_model_selection=args.meta_model,
                                  div_method=args.div_method)
            # 运行算法
            run_moeec_nsgaii_v5(problem=problem,
                                save_path=f'{save_path}/{exp_time_stamp}',
                                running_info=running_info,
                                fold=j)

    print(f'Total time: {(time.time()-start)/60:.2f}min')

