from typing import List, TypeVar
import yaml
import numpy as np
from .ensemble_model import EnsembleMKSVR, MultiKernelTransformer
from .solution import MOEESolution
S = TypeVar('S')

params = yaml.load(open('config/params.yml', encoding='utf8'), Loader=yaml.FullLoader)

class MOEEProblem:
    """ 
    多目标优化集成mkSVR问题
        - 数据集通过外部传入；
        - 数据集的处理方式：将全部数据集按比例划分为训练集、测试集；
        - 目标函数：精度、分散性、复杂度；
        - 可以指定编码方案：定长/变长（默认为变长）；
        - 可以指定meta集成模型；
        - 可以指定解的Diversity的计算方法；
    """

    def __init__(self, 
                 dataset_name: str, 
                 train_x, val_x, train_y, val_y,
                 fixed_length: int = -1, 
                 meta_model_selection: str = 'Linear',
                 div_method: str = 'NCL1'
                 ):
        self.dataset_name = dataset_name  # 记录数据集名称

        self.train_x, self.val_x, self.train_y, self.val_y = train_x, val_x, train_y, val_y  # 训练集&验证集
        
        self.fixed_length = fixed_length  # 指定染色体的长度（大于0的整数），默认为-1，表示长度不固定（变长编码方案）

        # 【预计算技巧(Precomputation technique)】在实例化problem类的同时实例化multi_kernel类, 而在multi_kernel的构造函数内则会调用函数求4个中间
        # 矩阵的计算结果，也就是说，这四个计算结果在一个problem优化问题中只计算一次，这样大大节省了运算量
        self.mk = MultiKernelTransformer(train_x=self.train_x, test_x=self.val_x)  # 计算多核输出类

        self.obj_labels = ['RMSE', 'Diversity', 'Complexity']  # 目标函数名称
        self.number_of_objectives = 3  # 目标函数维度

        if self.fixed_length > 0:
            # 定长编码：将MOEESolution中的参数上下界扩展成一个长向量
            self.para_bounds = {
                'low':  [0, 0,  2, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0] * self.fixed_length,  # 下界
                'high': [1, 10, 6, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1] * self.fixed_length  # 上界
            }
        else:
            # 变长编码：各参数上下界与MOEESolution中保持一致
            self.para_bounds = {
                'low':  [0, 0,  2, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],  # 下界
                'high': [1, 10, 6, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1]  # 上界
            }
        
        self.meta_model_selection = meta_model_selection  # 集成各子学习机的模型选择
        self.div_method = div_method  # 解的目标函数计算方法

    def create_solution(self) -> MOEESolution:
        """ 产生一个新解 """
        s = MOEESolution(fixed_length=self.fixed_length)  # 生成一个新解
        s = self.fix_solution(s)  # 修复解
        return s

    def fix_solution(self, solution: MOEESolution) -> MOEESolution:
        """ 修复解 """
        # 限制染色体长度/集成模型子学习机个数：最高40个子学习机，更多的话就随机删除
        max_length = params["solution_length_maximum"]
        if self.fixed_length == -1 and solution.k > max_length:
            # 注意：只有 self.fixed_length == -1 即变长编码方案下，才限制子学习机个数，定长编码时不做限制！
            solution.variables = solution.variables.sample(n=40)  # 随机选择40个子学习机进行保留
            solution.update_info()  # 更新/维护solution内部变量的信息

        # 将多项式核的degree参数转化为int型（参数矩阵中第3列）
        solution.variables['poly_degree'] = [round(i) for i in solution.variables['poly_degree']]

        # 归一化多核SVR的内部核函数权重
        w = solution.variables.loc[:, ['w1', 'w2', 'w3', 'w4', 'w5']].values
        solution.variables.loc[:, ['w1', 'w2', 'w3', 'w4', 'w5']] = w / np.sum(w, axis=1).reshape(-1, 1)

        # 归一化集成模型的集成权重
        W = solution.variables['W']
        solution.variables['W'] = W / W.sum()

        return solution
    
    def evaluate(self, population: List[MOEESolution]):
        """ 评价种群 """
        # 解码并训练种群中的所有解
        for solution in population:
            # 初始化集成模型
            solution.model = EnsembleMKSVR(paras=solution.variables, meta_model_selection=self.meta_model_selection)
            # 训练集成模型
            solution.model.fit(mk=self.mk, train_y=self.train_y)
        
        # 计算种群中所有解的三个目标函数
        for i, solution in enumerate(population):    
            # 获取Accuracy指标：RMSE
            RMSE = solution.model.get_RMSE(val_y=self.val_y)
            # 获取Diversity指标：DIV
            DIV = solution.model.get_DIV(method=self.div_method, pop=population, ind=i)
            # 获取Complexity指标：CMPLX
            CMPLX = solution.model.get_CMPLX()
            # 为解的三个目标函数赋值
            solution.objectives = [RMSE, DIV, CMPLX]         
    
    def get_name(self) -> str:
        return 'MOEE_Problem'

