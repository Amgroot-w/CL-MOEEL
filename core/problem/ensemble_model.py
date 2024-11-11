import time
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


class EnsembleMKSVR:
    """ 
    集成多核支持向量回归模型 (Ensemble Multi-kernel SVR)

        - 模型初始化：
            - 传入多个子mkSVR组成的参数矩阵（pd.DataFrame）；
            
            - 选择meta模型，用于集成各个子学习机的输出，默认为线性回归模型（Linear）；
                - 'WeightSum': 线性加权求和，与原编码方案一致（原默认的编码方案）;
                - 'Linear': 线性回归，在WightSum基础上，加了一个偏置项b（实验证明该方法效果最好，现设置为默认方案）;
                - 'MLP': BP神经网络;
                - 'SVR': 支持向量回归;

            - 选择Diversity指标的计算方法。依据解的分散性指标的定义方法的不同，分为以下两种情况：1) Diversity的评价是独立的，
            不依赖种群中的其他解；2) Diversity的评价不是独立的，依赖种群中的其他解。
                - 'VarianceSum': 方差和；
                - 'NCL1': 基于负相关学习-方法1；
                - 'NCL2': 基于负相关学习-方法2；
                - 'None': 去掉分散性指标，只保留剩下的两维目标函数；         
            
        - 模型训练：
            - 输入: problem对象中保存的mk（多核矩阵转换器）和标签值y；
            - 输出: 三个目标函数值（list型）；
    """

    def __init__(self, paras, meta_model_selection: str = 'Linear',
                 max_iter: int = -1, verbose: bool = False, print_output: bool = False):
        """ 初始化集成模型 """
        self.meta_model_selection = meta_model_selection  # meta模型选择
        self.paras = paras         # 编码在solution中的参数矩阵 (pd.DataFrame型)
        self.max_iter = max_iter   # LibSVM最大迭代次数，默认为-1表示不限制迭代次数
        self.verbose = verbose     # 是否打印展示输出
        self.print = print_output  # 是否打印输出子学习机训练进度
        self.k = len(paras)        # 获取mkSVR子学习机的个数 (1~20)

    def fit(self, mk, train_y):
        """ 
        集成模型训练
            输入的mk：Multi-kernel Transformer (多核矩阵转换器)
        """
        # 1.子学习机训练
        # 初始化k个mkSVR子学习机
        self.mkSVRs = [SVR(kernel='precomputed', verbose=self.verbose, max_iter=self.max_iter) for _ in range(self.k)]
        # 初始化k个mkSVR子学习机的预测输出值
        self.mkSVR_pred_train = [[] for _ in range(self.k)]  # 训练集预测输出
        self.mkSVR_pred_val = [[] for _ in range(self.k)]  # 验证集预测输出
        # 训练mkSVR子学习机
        for i in range(self.k):
            t1 = time.time()
            # 获取第i个mkSVR的参数
            mkSVR_paras = self.paras.loc['mkSVR{}'.format(i)]  
            # 计算训练集、验证集的格拉姆矩阵
            train_gram, val_gram = mk.transform(mkSVR_paras)
            # 训练mkSVR子学习机
            self.mkSVRs[i].fit(train_gram, train_y)
            # mkSVR子学习机预测
            self.mkSVR_pred_train[i] = self.mkSVRs[i].predict(train_gram)
            self.mkSVR_pred_val[i] = self.mkSVRs[i].predict(val_gram)
            # 打印输出
            t2 = time.time()
            if self.print:
                print('第{}个子学习机，degree参数={}，train_gram: shape={}, min={}, max={}, 用时：{:.2f}秒'.format(
                    i+1, mkSVR_paras[2], train_gram.shape, np.min(train_gram), np.max(train_gram), t2-t1))
        
        # 2.子学习机集成
        if self.meta_model_selection == 'WeightSum':
            # 使用进化算法优化的子学习机集成权重，来对子学习机做简单的线性加权求和
            self.meta_model = None  # 若采用原始的编码方案，则meta_model为设置为None
            self.ensemble_output = np.dot(self.paras['W'], np.array(self.mkSVR_pred_val))  # 计算集成模型的预测输出（验证集）
        
        else:
            # 训练一个独立的meta模型，来对各子学习机的输出做Stacking集成
            if self.meta_model_selection == 'Linear':
                self.meta_model = LinearRegression()  # 线性回归

            elif self.meta_model_selection == 'MLP':
                self.meta_model = MLPRegressor()  # 神经网络

            elif self.meta_model_selection == 'SVR':
                self.meta_model = SVR()  # 支持向量回归

            else:
                raise AttributeError('meta_model_selection input error!')
            
            # meta模型训练（训练集）
            self.meta_model.fit(np.array(self.mkSVR_pred_train).T, train_y)
            # meta模型预测（验证集）
            self.ensemble_output = self.meta_model.predict(np.array(self.mkSVR_pred_val).T)

    def get_RMSE(self, val_y):
        """
        获取Accuracy指标：均方根误差RMSE（最小化）
        """
        val_pred = self.ensemble_output  # 获取集成模型预测输出
        RMSE = root_mean_squared_error(val_y, val_pred)  # 这里不平方，表示均方根误差
        return RMSE
    
    def get_DIV(self, method, pop, ind):
        """
        获取Diversity指标：DIV（最小化）
            :param method: str，选择DIV的计算方法
            :param pop: List，种群
            :param ind: int，解的索引
        """

        def ComputeNCL(arr: np.array, ind: int):
            """
            负相关学习（Negative correlation learning, NCL）
                :param arr: np.array，数组
                :param ind: int，索引
                :return ncl: np.float64，数组arr中第ind列的NCL值
            
                注：返回值ncl越小，表明ind列与其他列的差异越大！
            """
            # 原论文中的定义公式（Ensemble learning via negative correlation-Yong Liu/Xin Yao）
            # error = arr - np.mean(arr, axis=1, keepdims=True)
            # target = error[:, ind]
            # others = np.delete(error, ind, axis=1)
            # ncl = np.sum(target * np.sum(others, axis=1))

            # 推导之后的定义公式（采用负相关学习的SVM集成算法-洪铭/汪鸿翔/刘晓芳/柳培忠）
            error = arr[:, ind] - np.mean(arr, axis=1)
            ncl = -1.0 * np.sum(error**2)

            return ncl

        if method == 'VarianceSum':
            # 编码矩阵的方差和
            # 不算最后一列的方差，加负号变为最小化
            DIV = -1.0 * self.paras.iloc[:, :12].var().sum()
        
        elif method == 'NCL1':
            # 基于负相关学习NCL：方法1（单个解内部独立计算）
            # 获取集成模型各个子学习机的预测输出，计算每个子学习机的NCL值，对其取平均得到DIV指标值
            preds = np.array(self.mkSVR_pred_val).T
            # DIV = -1.0 * np.mean([ComputeNCL(preds, i) for i in range(preds.shape[1])])
            DIV = np.mean([ComputeNCL(preds, i) for i in range(preds.shape[1])])

        elif method == 'NCL2':
            # 基于负相关学习NCL：方法2（每个解都依赖种群中其他解来计算）
            # 获取种群中所有解对应的集成模型的输出，计算该解对应的集成模型的NCL值，作为DIV指标值
            preds = np.array([s.model.ensemble_output for s in pop]).T
            # DIV = -1.0 * ComputeNCL(preds, ind)
            DIV = ComputeNCL(preds, ind)

        elif method == 'None':
            # 去掉Diversity指标，只保留RMSE和Complexity两个目标函数
            # 直接将DIV设置为固定值0
            DIV = 0

        else:
            raise AttributeError('Input method error!')
        
        return DIV

    def get_CMPLX(self):
        """
        获取Complexity指标：支持向量个数平均值CMPLX（最小化）
        """
        # n_supports = [self.mkSVRs[i].n_support_[0] for i in range(self.k)]  # 提取支持向量的个数
        # self.Complexity = np.mean(n_supports) + max(n_supports)-min(n_supports)  # 支持向量个数均值 + 惩罚项（最大减最小）
        CMPLX = np.mean([self.mkSVRs[i].n_support_[0] for i in range(self.k)])  # 支持向量个数均值
        return CMPLX

    def predict(self, mk, test_x):
        """ 集成模型预测 """
        # 初始化k个mkSVR子学习机的预测输出值
        mkSVR_pred = [[] for _ in range(self.k)]
        for i in range(self.k):
            # 获取第i个mkSVR的参数
            mkSVR_paras = self.paras.loc['mkSVR{}'.format(i)]  
            # 计算测试数据的格拉姆矩阵
            test_gram = mk.transform_other_data(mkSVR_paras, test_x)
            # 第i个mkSVR子学习机预测
            mkSVR_pred[i] = self.mkSVRs[i].predict(test_gram)
        
        if self.meta_model_selection == 'WeightSum':
            # 原始编码方案线性加权求和（测试集）
            test_pred = np.dot(self.paras['W'], np.array(mkSVR_pred))
        else:
            # meta模型预测输出（测试集）
            test_pred = self.meta_model.predict(np.array(mkSVR_pred).T)

        return test_pred


# 多核矩阵转换器
class MultiKernelTransformer:
    """
    实现了多核方法 
        --- 预计算技巧 (Precomputation technique)
    
    主要功能：
        计算训练集(train_x)和测试集(test_x)在希尔伯特空间的高维映射，以便于在外部mkSVR模型拟合的时
    候直接传入这个高维矩阵来训练模型（注意：此处说的测试集并不是进化算法的测试集，而是对应进化算法
    中的验证集）。

    **注意：实例化该类的时候，直接传入训练集和测试集数据保存在类内部，计算高维矩阵时，再根据传入的
    paras参数来计算，这样可以预先计算好很多可以共用的矩阵运算结果，从而提高程序运行效率。

    程序优化：
        1. 通过传入precomputed的temp，节省大量的矩阵运算时间；
        2. 计算rbf核的gram矩阵时，由于它是半正定的，只计算矩阵上三角即可(仅限于训练集，测试集仍需
        计算全部，因为它不是对称的)。

    """

    def __init__(self, train_x, test_x, precomputed=True):
        self.train_x = train_x  # 训练集x
        self.test_x = test_x  # 测试集x

        # 默认进行预计算（用于进化过程）
        self.precomputed = precomputed
        if self.precomputed:
            # 预计算矩阵运算共用的中间值1：矩阵点积（格拉姆矩阵）: <x, x'>, 该项被线性核、多项式核、sigmoid核共用
            self.temp1_train = self.ComputeTemp1(train_x, train_x)  # 训练集：本身的点积
            self.temp1_test = self.ComputeTemp1(test_x, train_x)  # 测试集：与训练集的的点积 (注意test_x写在前面)

            # 预计算矩阵运算共用的中间值2：矩阵向量的欧氏距离（向量差的二范数）: ||x - x'||, 该项被高斯 v核、拉普拉斯核共用
            self.temp2_train = self.ComputeTemp2(train_x, train_x, 'train')  # 训练集：与本身向量的欧式距离
            self.temp2_test = self.ComputeTemp2(test_x, train_x, 'test')  # 测试集：与训练集向量的的欧氏距离 (注意test_x写在前面)
        
    @staticmethod
    def ComputeTemp1(x1, x2):
        """ 预计算中间变量temp1 """
        return np.dot(x1, x2.T)

    @staticmethod
    def ComputeTemp2(x1, x2, label):
        """ 预计算中间变量temp2 """
        m1, m2 = len(x1), len(x2)  # 两个矩阵各自的样本数
        distance = np.zeros((m1, m2))  # 初始化矩阵
        if label == 'train':
            # 遍历矩阵x1的行向量（一行为一条样本）
            for i in range(m1):
                v1 = x1[i]
                # 遍历矩阵x2的行向量（一行为一条样本）
                for j in range(i, m2):
                    v2 = x2[j]
                    # 计算两个向量的二范数
                    norm = np.linalg.norm(v1 - v2, ord=2)
                    # 存储在半正定核矩阵中对称的位置
                    distance[i, j] = norm
                    distance[j, i] = norm
        else:
            # 当数据集是test或val时，由于格拉姆矩阵不是正定的，因此只能计算全部矩阵，而不是只计算上三角
            # 遍历矩阵x1的行向量（一行为一条样本）
            for i in range(m1):
                v1 = x1[i]
                # 遍历矩阵x2的行向量（一行为一条样本）
                for j in range(m2):
                    v2 = x2[j]
                    # 计算两个向量的二范数
                    distance[i, j] = np.linalg.norm(v1 - v2, ord=2)

        return distance

    @staticmethod
    def linear_kernel(x1, x2, precomputed=False, temp=None):
        """ 1.线性核，无参数 """
        if precomputed:
            return temp
        else:
            return np.dot(x1, x2.T)

    @staticmethod
    def polynomial_kernel(x1, x2, gamma, coef0, degree, precomputed=False, temp=None):
        """ 2.多项式核，参数：gamma, coef0, degree """
        if precomputed:
            return (gamma*temp + coef0)**degree
        else:
            return (gamma*np.dot(x1, x2.T) + coef0)**degree

    @staticmethod
    def rbf_kernel(x1, x2, gamma, precomputed=False, temp=None):
        """ 3.高斯核，参数：gamma """
        if precomputed:
            return np.exp(-gamma * temp ** 2)
        else:
            distance = np.zeros((x1.shape[0], x2.shape[0]))
            # 遍历矩阵x1、x2的行向量（一行为一条样本）
            for (i, v1) in enumerate(x1):
                for (j, v2) in enumerate(x2):
                    distance[i, j] = np.linalg.norm(v1 - v2, ord=2)
            return np.exp(-gamma * distance ** 2)

    @staticmethod
    def laplace_kernel(x1, x2, gamma, precomputed=False, temp=None):
        """ 4.拉普拉斯核，参数：gamma """
        if precomputed:
            return np.exp(-gamma * temp)
        else:
            distance = np.zeros((x1.shape[0], x2.shape[0]))
            # 遍历矩阵x1、x2的行向量（一行为一条样本）
            for (i, v1) in enumerate(x1):
                for (j, v2) in enumerate(x2):
                    distance[i, j] = np.linalg.norm(v1 - v2, ord=2)
            
            return np.exp(-gamma * distance)

    @staticmethod
    def sigmoid_kernel(x1, x2, gamma, coef0, precomputed=False, temp=None):
        """ 5.Sigmoid核，参数：gamma, coef0 """
        if precomputed:
            return np.tanh(gamma * temp + coef0)
        else:
            return np.tanh(gamma * np.dot(x1, x2.T) + coef0)

    def linear_ensemble_kernels(self, x1, x2, paras, precomputed=False, temp1=None, temp2=None):
        """
        多核线性加权集成 
        """
        kernels = np.array([
            # 1.线性核
            self.linear_kernel(x1, x2, 
                precomputed=precomputed, temp=temp1
            ),
            # 2.多项式核
            self.polynomial_kernel(x1, x2, 
                gamma=paras['poly_gamma'], coef0=paras['poly_coef0'], degree=paras['poly_degree'], 
                precomputed=precomputed, temp=temp1
            ),
            # 3.高斯核
            self.rbf_kernel(x1, x2, 
                gamma=paras['rbf_gamma'], 
                precomputed=precomputed, temp=temp2
            ),
            # 4.拉普拉斯核
            self.laplace_kernel(x1, x2, 
                gamma=paras['laplace_gamma'], 
                precomputed=precomputed, temp=temp2
            ),
            # 5.Sigmoid核
            self.sigmoid_kernel(x1, x2, 
                gamma=paras['sigmoid_gamma'], coef0=paras['sigmoid_coef0'], 
                precomputed=precomputed, temp=temp1
            )
        ])

        weights = np.array([paras['w1'], paras['w2'], paras['w3'], paras['w4'], paras['w5']]).reshape(1, -1)

        res = np.dot(
            weights, 
            kernels.reshape(-1, kernels.shape[1]*kernels.shape[2]), 
        ).reshape(kernels.shape[1], kernels.shape[2])

        return res

    def transform(self, paras):
        """
        转换训练集、测试集
            按照paras参数将train_x和test_x映射到高维的希尔伯特空间（用于进化过程，减少计算量）
        """
        train_gram = self.linear_ensemble_kernels(
            self.train_x, self.train_x, paras, 
            precomputed=True, temp1=self.temp1_train, temp2=self.temp2_train
        )  # 映射训练集
        
        test_gram = self.linear_ensemble_kernels(
            self.test_x, self.train_x, paras, 
            precomputed=True, temp1=self.temp1_test, temp2=self.temp2_test
        )  # 映射测试集

        return train_gram, test_gram
    
    def transform_other_data(self, paras, x):
        """ 
        转换其他任意数据
            按照paras参数将x映射到高维的希尔伯特空间（x必须与self.train_x具有相同的维度，用于在线测试）
        """
        if self.precomputed:
            x_gram = self.linear_ensemble_kernels(
                x, self.train_x, paras, 
                precomputed=True, temp1=self.temp1_test, temp2=self.temp2_test)
        else:
            x_gram = self.linear_ensemble_kernels(
                x, self.train_x, paras)
        
        return x_gram


