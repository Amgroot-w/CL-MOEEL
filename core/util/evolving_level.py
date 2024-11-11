from abc import abstractmethod

"""
定义了进化程度
"""

class EvolvingLevel:
    """ 
    进化程度 - 父类

        - 以level表示进化程度;
        - level范围是[0, 1], 0表示还未进化, 1表示进化完成.
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute(self) -> float:
        """ 计算进化程度 """
        pass


class EpochBasedEvolvingLevel(EvolvingLevel):
    """ 基于Epoch的进化程度 """
    
    def __init__(self, max_epoch: int):
        self.max_epoch = max_epoch  # 最大进化代数

    def compute(self, epoch) -> float:
        """ 计算进化程度 """
        return epoch / self.max_epoch



class PFDistanceBasedEvolvingLevel(EvolvingLevel):
    """ 基于PF距离的进化程度 """

    def __init__(self):
        pass  

    def compute(self) -> float:
        """ 计算进化程度 """
        pass

