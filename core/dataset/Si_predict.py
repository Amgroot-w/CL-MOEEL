import copy
import yaml
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .MyDataset import MyDatasetKFold

params = yaml.load(open('config/params.yml', encoding='utf8'), Loader=yaml.FullLoader)

"""
铁水硅含量预测数据集
"""

class Si_predict(MyDatasetKFold):
    def __init__(self):
        super().__init__(random_state=0)
        self.number_of_samples = self.raw_data.shape[0]  # 样本数
        self.number_of_features = self.raw_data.shape[1] - 1  # 特征数
        self.hr_threshold = 0.1  # 指定HR指标的计算阈值
        self.INFO = '**选作Si_predict数据集**'  # 数据集说明
    
    def read_data(self):
        df = pd.read_excel('data/Silicon Content Prediction Data Set/Si_predict.xlsx', header=None)
        return df

    def preprocess(self, data, scaler):
        """ 数据预处理 """
        preprocessed_data = copy.deepcopy(data)

        data_y = preprocessed_data.iloc[:, -1].values  # 最后一列为硅含量标签
        data_x = preprocessed_data.iloc[:, :-1].values  # 前面67列为输入变量
        
        return data_x, data_y, scaler

    def get_name(self):
        return 'Si_predict'    

