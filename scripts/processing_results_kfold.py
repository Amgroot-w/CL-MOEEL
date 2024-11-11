import os
import sys; sys.path.append(os.getcwd())
import yaml
import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from core.problem import MultiKernelTransformer
from core.util.front import get_non_dominated_solutions
from core.util.select_solution import KneePointSelector
from core.dataset import get_dataset
from utils.indicators import HitRate

"""
整理MOEE重复实验结果 -- K折交叉验证版本
"""

params = yaml.load(open('config/params.yml', encoding='utf8'), Loader=yaml.FullLoader)

parser = argparse.ArgumentParser('Processing MOEE results -- kfold')
parser.add_argument('--dataset', type=str, default='', help="Choose dataset")
parser.add_argument('--path', type=str, default='', help="MOEE model path")
args = parser.parse_args()  # 解析参数

dataset = get_dataset(args.dataset)  # 数据集

model_path = f'models/{dataset.get_name()}/{args.path}'  # 保存MOEE重复实验结果的根目录

indicators = {}  # 记录重复实验指标结果

for exp_name in os.listdir(model_path):
    dirname = f"{model_path}/{exp_name}"
    if os.path.isfile(dirname):
        continue

    for j, file_name in enumerate(os.listdir(dirname)):
        file_dir = f'{dirname}/{file_name}'
        if os.path.isfile(file_dir):
            continue

        print(f'Processing: {exp_name}, {file_dir}')

        # 读取进化结果
        with open(file_dir + '/results.pkl', 'rb') as f:
            results = pickle.load(f)  # 读取进化完成后的种群

        with open(file_dir + '/log.pkl', 'rb') as f:
            log = pickle.load(f)  # 读取进化过程信息

        front = get_non_dominated_solutions(solutions=results)  # 获取Pareto前沿
        
        solution_selector = KneePointSelector(front=front)  # 选解方法：膝点法

        theChosenOne = solution_selector.execute()  # 选解

        train_x, _, test_x, _, _, test_y = dataset.get_split_data(method='kfold', fold=j)
        mk = MultiKernelTransformer(train_x, test_x, precomputed=True)

        pred = theChosenOne.model.predict(mk, test_x)  # 计算在测试集上的模型输出
        true = test_y  # 测试集真实值

        # 模型预测输出
        pd.DataFrame({'Pred': pred, 'True': true}).to_csv(f'{file_dir}/preds.csv', index=False)

        # 计算评价指标
        rmse = root_mean_squared_error(true, pred)  # 均方根误差RMSE
        r2 = r2_score(true, pred)  # 决定系数R2
        hr = HitRate(true, pred, threshold=dataset.hr_threshold)  # 命中率HR

        # 记录一下本次进化算法得到的非支配解个数
        size_pf = len(front)  # 非支配解个数
        size_pop = len(results)  # 种群大小

        # 记录算法运行时间
        runtime = log['runtime']

        # 记录最终选择的解的子学习机个数
        length = theChosenOne.k

        indicators[file_name] = [exp_name, j, rmse, r2, hr, size_pf, size_pop, runtime, length]


# 保存所有实验结果
indicators = pd.DataFrame(indicators, index=['exp', 'kfold', 'RMSE', 'R2', 'HR', 'PF', 'Pop', 'Runtime', 'Length']).T
indicators.to_csv(f'{model_path}/results.csv')

# 计算15次重复实验的结果指标
inds = indicators.groupby('exp').mean().drop(columns='kfold')
pd.DataFrame(inds).to_csv(f'{model_path}/inds.csv')

