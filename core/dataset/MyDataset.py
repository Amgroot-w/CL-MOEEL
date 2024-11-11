from abc import abstractmethod
import yaml
from sklearn.model_selection import train_test_split, KFold

params = yaml.load(open('config/params.yml', encoding='utf8'), Loader=yaml.FullLoader)

class MyDatasetKFold:
    """ 
    数据集父类 -- K折交叉验证
    """

    def __init__(self, random_state: int):
        self.random_state = random_state  # 随机数种子
        self.raw_data = self.read_data()  # 读取原始数据
        self.kfold_indexes = self.kfold_split()  # K折交叉验证拆分数据集
        self.number_of_samples = None  # 处理后的样本数
        self.number_of_features = None  # 处理后的特征数
        self.hr_threshold = 'auto'  # 数据集的HR指标阈值，默认为'auto'自动计算
        self.INFO = ''  # 数据集说明（此处的文本会自动记录在docs\DatasetInfo.md中）

    @abstractmethod
    def read_data(self):
        pass
    
    def kfold_split(self):
        # K折交叉验证---先随机打乱，然后拆分为k份，返回数据索引
        kf = KFold(n_splits=params['kfold_n_splits'], shuffle=True, random_state=self.random_state)
        kfold_indexes = list(kf.split(self.raw_data))
        return kfold_indexes
    
    @abstractmethod
    def preprocess(self, data, scaler):
        """ 数据预处理 """
        pass

    def get_split_data(self, method: str = 'kfold', fold: int = None, val: bool = True):
        """ K折交叉验证-拆分数据集 """
        if method != 'kfold':
            raise AttributeError(f"Input method is wrong: {method}, must be kfold!")
        
        # 获取第fold折的训练集、测试集数据
        train_index, test_index = self.kfold_indexes[fold][0], self.kfold_indexes[fold][1]
        self.train_data, self.test_data = self.raw_data.iloc[train_index, :], self.raw_data.iloc[test_index, :]

        # 数据预处理
        self.train_x, self.train_y, scaler = self.preprocess(data=self.train_data, scaler=None)  # 数据预处理：训练集
        self.test_x, self.test_y, scaler = self.preprocess(data=self.test_data, scaler=scaler)  # 数据预处理：测试集

        if val:
            # 从训练集中再拆分出验证集
            self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(
                self.train_x, self.train_y, test_size=1/(params['kfold_n_splits']-1), shuffle=False, random_state=self.random_state)
            
            return self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y
        
        else:
            # 直接返回拆分好的训练集、测试集
            return self.train_x, self.test_x, self.train_y, self.test_y
        
    @abstractmethod
    def get_name(self):
        pass

