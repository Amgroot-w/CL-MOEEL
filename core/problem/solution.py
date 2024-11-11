import random
import pandas as pd
import yaml

params = yaml.load(open('config/params.yml', encoding='utf8'), Loader=yaml.FullLoader)

class MOEESolution:
    """
    1.编码方案：
    -----------------------------------------------------------------------------------------------------------------
    |                          参数部分                               |                       权重部分                |
    -----------------------------------------------------------------------------------------------------------------
    |        多项式核          |  高斯核 |  拉普拉斯核 |    Sigmoid核     |            核函数权重              | 集成权重 |
    |  poly    poly    poly   |   rbf  |   laplace  | sigmoid sigmoid |  w1     w2     w3     w4     w5   |   W     |
    ----------------------------------------------- -----------------------------------------------------------------
    | gamma   coef0   degree  |  gamma |    gamma   |   gamma  coef0  | w_11   w_12   w_13   w_14   w_15  |  W_1     |
    | gamma   coef0   degree  |  gamma |    gamma   |   gamma  coef0  | w_21   w_22   w_23   w_24   w_25  |  W_2     |
    |          .              |        |     .      |                 |                      .           |          |
    |          .              |        |     .      |                 |                      .           |          |
    |          .              |        |     .      |                 |                      .           |          |
    | gamma   coef0   degree  |  gamma |    gamma   |   gamma  coef0  | w_k1   w_k2   w_k3   w_k4   w_k5  |  W_k    |
    ----------------------------------------------------------------------------------------------------------------
    下界：0     0       2          0          0          0       0       0       0      0      0      0       0     |
    上界：1     10      6          1          1          1       10      1       1      1      1      1       1     |
    ----------------------------------------------------------------------------------------------------------------
    个体解solution用pd.DataFrame实现，df的列名(df.columns)为各个参数的名称，df的行索引(df.index)为mkSVR子学习机的编号.

    2. 目标函数：
        Objective #1 -- 精度(Accuracy): 均方根误差损失RMSE--- 最小化
        Objective #2 -- 分散性(Diversity): 基于方差和计算得到--- 最小化
        Objective #3 -- 复杂度(Complexity): 所有mkSVR子学习机的支持向量个数和/均值 --- 最小化
    
    3. 可选编码方案：变长/定长
        - 变长（默认）：self.k从整数1~20中随机取值，作为解的初始长度；
        - 定长：self.k初始化时就固定为10，且后续进化过程中不会再改变；
    """

    def __init__(self, fixed_length: int = -1):
        # fixed_length: 指定染色体的长度（大于0的整数），默认为-1，表示长度不固定（变长编码方案）

        # 参数名称规定
        self.para_names = [
            'poly_gamma', 'poly_coef0', 'poly_degree',  # 多项式核内部参数
            'rbf_gamma',  # 高斯核内部参数
            'laplace_gamma',  # 拉普拉斯核内部参数
            'sigmoid_gamma', 'sigmoid_coef0',  # Sigmoid核内部参数
            'w1', 'w2', 'w3', 'w4', 'w5',  # 核函数权重
            'W'  # 集成权重
        ]

        # 各参数的上下界（与MOEEProblem中保持一致）
        self.para_bounds = {
            'low': [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 下界
            'high': [1, 10, 6, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1]  # 上界
        }

        # 个体解变量矩阵的维度，即等于参数个数（7个核函数参数 + 5个核函数权重 + 1个集成权重 = 13维）
        # 注意：该值在整个进化过程中固定不变，也就是说mkSVR子学习机的组成结构是固定的
        self.n = len(self.para_names)  
       
        # 确定个体解的编码方案：变长/定长
        if fixed_length > 0:
            # 定长编码：染色体长度固定不变
            self.k = fixed_length
        else:
            # 变长编码：从1~20中随机选择一个整数，作为新个体的随机初始长度
            lower_bound = params["solution_length_init_lowerbound"]
            upper_bound = params["solution_length_init_upperbound"]
            self.k = random.randint(lower_bound, upper_bound)
            
        # 生成随机参数矩阵(二维列表)
        para_matrix = [
            [random.uniform(
                self.para_bounds['low'][i], self.para_bounds['high'][i]
            ) for i in range(self.n)] for _ in range(self.k)
        ]
        # 赋值给个体解，并转化为DataFrame格式
        self.variables = pd.DataFrame(
            para_matrix,
            columns=self.para_names,
            index = ['mkSVR{}'.format(i) for i in range(self.k)]
        )
        
        # 初始化个体解对应的模型
        self.model = None  # 该model变量初始化为None，并在评价个体解时进行更新

        # 初始化个体解的目标函数
        self.objectives = list([None, None, None])
        self.number_of_objectives = len(self.objectives)

        # 初始化个体解的属性
        self.attributes = {}

    def update_info(self):
        """ 更新/维护solution内部变量的信息 """
        # 更新染色体长度
        self.k = len(self.variables)
        # 更新DataFrame的index
        self.variables.index = ['mkSVR{}'.format(i) for i in range(self.k)]
    
    def replace_by(self, s):
        """ 将另一个解赋值给该解，但地址保持不变 """
        self.k = s.k
        self.variables = s.variables
        self.attributes = s.attributes
        self.model = s.model
        self.objectives = s.objectives
    
    @staticmethod
    def Pop2DataFrame(population):
        """
        提取DataFrame类型的种群（外部调用方法） 
            转换: List[MOEESolution] --> pd.DataFrame
        """
        pop = pd.DataFrame(
            [solution.objectives for solution in population], 
            columns=['RMSE', 'Diversity', 'Complexity'])
        return pop

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self) -> str:
        return 'MOEESolution(mkSVR_num={}, objectives={})'.format(self.k, self.objectives)

