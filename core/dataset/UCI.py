import copy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .MyDataset import MyDatasetKFold

"""
UCI系列数据集
"""

class UCI14_ResidentialBuilding(MyDatasetKFold):
    """
    UCI14数据集 -- K折交叉验证
    """

    def __init__(self):
        super().__init__(random_state=0)
        self.number_of_samples = self.raw_data.shape[0]
        self.number_of_features = self.raw_data.shape[1] - 2
        self.INFO = '**选作Benchmark数据集**'  # 数据集说明

    def read_data(self):
        raw_data = pd.read_excel('data/Residential Building Data Set/Residential-Building-Data-Set-preprocessed.xlsx', header=None)
        return raw_data
        
    def preprocess(self, data, scaler):
        """ 数据预处理 """
        preprocessed_data = copy.deepcopy(data)

        # 归一化
        if scaler == None:
            scaler = MinMaxScaler()
            preprocessed_data.iloc[:, :] = scaler.fit_transform(preprocessed_data.iloc[:, :])
        else:
            preprocessed_data.iloc[:, :] = scaler.transform(preprocessed_data.iloc[:, :])

        data_x = preprocessed_data.iloc[:, :-2].values
        data_y = preprocessed_data.iloc[:, -2].values  # 选择V-9（Actual sales prices）作为输出特征
        
        return data_x, data_y, scaler
        
    def get_name(self):
        return 'UCI14_ResidentialBuilding'

