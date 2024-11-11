from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import random
from cv2 import threshold

S = TypeVar('S')

class Comparator(Generic[S], ABC):
    """ 比较器 """
    @abstractmethod
    def compare(self, solution1: S, solution2: S) -> int:
        """
        规定：
        传入两个solution，它们必须有objectives属性（list类型）且已经被赋值
        传出int型比较结果，有三种情况：-1, 0, 1, 分别是：
            s1 优于 s2: result = -1;
            s2 优于 s1: result = +1;
            s1与s2一致: result = 0
        """
        pass

class SlackComparator(Comparator):
    """
    松弛比较器
        用于松弛二元锦标赛选择算子

    目标函数#1 (精度)：RMSE
    目标函数#2 (多样性)：DIV
    目标函数#3 (复杂度)：CMPLX

        松弛比较的逻辑是：先比较RMSE (obj_1)，并直接返回RMSE显著更低的解；若在设定的阈值范围内判定两个解的RMSE
    无显著差别时，再比较Complexity (obj_2)，并返回复杂度更低的解；若在设定的阈值范围内判定两个解的Complexity
    也没有显著差别时，最后再比较Diversity (obj_3)，并返回Diversity更高的解；若Diversity指标也相同，则认为两
    个解无显著优劣之分，并返回0. (优先级：obj_1 -> obj_2 -> obj_3)

    """

    def init_threshold(self, threshold1, threshold2, threshold3):
        """
        初始化三个目标函数的阈值
            注意这三个阈值的设置和问题有关，比如硅含量预测问题和气温预测的RMSE就不在同一个数量级上，所以需要在
            初始化进化种群并评价初始种群后，调用该方法自动地设置三个阈值
        """
        self.threshold1 = threshold1  # RMSE的阈值
        self.threshold2 = threshold2  # DIV的阈值
        self.threshold3 = threshold3  # CMPLX的阈值

    def update_threshold(self, threshold1, threshold2, threshold3):
        """
        更新目标函数的阈值
            该函数用于在每个epoch中，基于pop所有个体的目标函数值，对三个目标函数的阈值进行更新
        """
        self.threshold1 = threshold1  # RMSE的阈值
        self.threshold2 = threshold2  # DIV的阈值
        self.threshold3 = threshold3  # CMPLX的阈值

    def compare(self, solution1: S, solution2: S, prefer_obj: str = 'RMSE') -> int:
        # prefer_obj: 设置偏好的目标函数，默认为RMSE

        if solution1 is None:
            raise Exception("The solution1 is None")

        elif solution2 is None:
            raise Exception("The solution2 is None")

        if prefer_obj == 'RMSE':  # 侧重于RMSE，优先级：RMSE -> Complexity -> Diversity
            obj_1a, obj_1b = solution1.objectives[0], solution2.objectives[0]  # 优先级最高的目标函数
            obj_2a, obj_2b = solution1.objectives[2], solution2.objectives[2]  # 优先级次之的目标函数
            obj_3a, obj_3b = solution1.objectives[1], solution2.objectives[1]  # 优先级最低的目标函数
            alpha, beta = self.threshold1, self.threshold3

        elif prefer_obj == 'Diversity':  # 侧重于Diversity，优先级：Diversity -> RMSE -> Complexity
            obj_1a, obj_1b = solution1.objectives[1], solution2.objectives[1]  # 优先级最高的目标函数
            obj_2a, obj_2b = solution1.objectives[0], solution2.objectives[0]  # 优先级次之的目标函数
            obj_3a, obj_3b = solution1.objectives[2], solution2.objectives[2]  # 优先级最低的目标函数
            alpha, beta = self.threshold2, self.threshold1

        elif prefer_obj == 'Complexity':  # 侧重于Complexity，优先级：Complexity -> RMSE -> Diversity
            obj_1a, obj_1b = solution1.objectives[2], solution2.objectives[2]  # 优先级最高的目标函数
            obj_2a, obj_2b = solution1.objectives[0], solution2.objectives[0]  # 优先级次之的目标函数
            obj_3a, obj_3b = solution1.objectives[1], solution2.objectives[1]  # 优先级最低的目标函数
            alpha, beta = self.threshold3, self.threshold1
        
        else:
            raise AttributeError(f'Input prefer_obj is invalid: {prefer_obj}')
        
        # 开始比较
        if abs(obj_1a - obj_1b) <= alpha:
            if abs(obj_2a - obj_2b) <= beta:
                if obj_3a < obj_3b:
                    return -1
                elif obj_3a > obj_3b:
                    return 1
                else:
                    return 0
            else:
                if obj_2a < obj_2b:
                    return -1
                else:
                    return 1
        else:
            if obj_1a < obj_1b:
                return -1
            else:
                return 1


class SlackComparator_v1(SlackComparator):
    """
    松弛比较器-改进版本1

    改进后的比较逻辑是：
        1. 先比较偏好目标函数，并返回函数值更优的解；
        2. 若两个解在偏好目标函数上的差异小于阈值，则返回随机解；
    """

    def compare(self, solution1: S, solution2: S, prefer_obj: str = 'RMSE') -> int:
        # prefer_obj: 设置偏好的目标函数，默认为RMSE

        if solution1 is None:
            raise Exception("The solution1 is None")

        elif solution2 is None:
            raise Exception("The solution2 is None")

        if prefer_obj == 'RMSE':
            f1, f2 = solution1.objectives[0], solution2.objectives[0]
            alpha = self.threshold1

        elif prefer_obj == 'Diversity':
            f1, f2 = solution1.objectives[1], solution2.objectives[1]
            alpha = self.threshold2

        elif prefer_obj == 'Complexity':
            f1, f2 = solution1.objectives[2], solution2.objectives[2]
            alpha = self.threshold3

        else:
            raise AttributeError(f'Input prefer_obj is invalid: {prefer_obj}')
        
        # 开始比较
        if abs(f1 - f2) <= alpha:
            # 两个解在偏好目标函数上的差异在阈值范围内，则认为两个解是同等优秀的
            return 0
        else:
            # 否则，返回在偏好目标函数上更有优势的解
            if f1 < f2:
                return -1
            else:
                return 1


class SlackComparator_v2(SlackComparator):
    """
    松弛比较器-改进版本2

    改进后的比较逻辑是：
        1. 先比较偏好目标函数，并返回函数值更优的解；
        2. 若两个解在偏好目标函数上的差异小于阈值，则在比较两个解的支配关系：
            (1) 若其中一个解支配另一个解，则返回非支配解；
            (2) 若两个解互为非支配，则返回随机解；
    """

    def compare(self, solution1: S, solution2: S, prefer_obj: str = 'RMSE') -> int:
        # prefer_obj: 设置偏好的目标函数，默认为RMSE

        if solution1 is None:
            raise Exception("The solution1 is None")

        elif solution2 is None:
            raise Exception("The solution2 is None")

        if prefer_obj == 'RMSE':
            f1, f2 = solution1.objectives[0], solution2.objectives[0]
            alpha = self.threshold1

        elif prefer_obj == 'Diversity':
            f1, f2 = solution1.objectives[1], solution2.objectives[1]
            alpha = self.threshold2

        elif prefer_obj == 'Complexity':
            f1, f2 = solution1.objectives[2], solution2.objectives[2]
            alpha = self.threshold3

        else:
            raise AttributeError(f'Input prefer_obj is invalid: {prefer_obj}')
        
        # 开始比较
        if abs(f1 - f2) <= alpha:
            # 两个解在偏好目标函数上的差异在阈值范围内，则比较两个解的支配关系
            return DominanceComparator().compare(solution1, solution2)
        else:
            # 否则，返回在偏好目标函数上更有优势的解
            if f1 < f2:
                return -1
            else:
                return 1


class DominanceComparator(Comparator):
    """ 
    基于支配关系的比较
        比较对象：解的目标函数（多维）
        默认目标函数值越小越好
    """

    def compare(self, solution1: S, solution2: S) -> int:
        if solution1 is None:
            raise Exception("The solution1 is None")
        elif solution2 is None:
            raise Exception("The solution2 is None")

        vector1, vector2 = solution1.objectives, solution2.objectives

        result = 0
        for i in range(len(vector1)):
            if vector1[i] > vector2[i]:
                if result == -1:
                    return 0
                result = 1

            elif vector2[i] > vector1[i]:
                if result == 1:
                    return 0
                result = -1

        return result


class SolutionAttributeComparator(Comparator):
    """
    基于属性值大小的比较
        比较对象：解的某个属性值（一维）
        默认属性值越小越好
    """

    def __init__(self, key: str, lowest_is_best: bool = True):
        self.key = key  # 要比较的解的属性名称（字典类型的键值）
        self.lowest_is_best = lowest_is_best  # 属性值越小表示解越优秀

    def compare(self, solution1: S, solution2: S) -> int:
        value1 = solution1.attributes.get(self.key)
        value2 = solution2.attributes.get(self.key)

        result = 0
        if value1 is not None and value2 is not None:
            if self.lowest_is_best:
                if value1 < value2:
                    result = -1
                elif value1 > value2:
                    result = 1
                else:
                    result = 0
            else:
                if value1 > value2:
                    result = -1
                elif value1 < value2:
                    result = 1
                else:
                    result = 0

        return result


class SolutionObjectiveComparator(Comparator):
    """
    基于目标函数值大小的比较
        比较对象：解的某个目标函数值
        默认目标函数值值越小越好  
    """

    def __init__(self, obj_index: int, lowest_is_best: bool = True):
        self.obj_index = obj_index  # 要比较的解的目标函数值索引
        self.lowest_is_best = lowest_is_best  # 目标函数值越小表示解越优秀

    def compare(self, solution1: S, solution2: S) -> int:
        value1 = solution1.objectives[self.obj_index]
        value2 = solution2.objectives[self.obj_index]

        result = 0
        if value1 is not None and value2 is not None:
            if self.lowest_is_best:
                if value1 < value2:
                    result = -1
                elif value1 > value2:
                    result = 1
                else:
                    result = 0
            else:
                if value1 > value2:
                    result = -1
                elif value1 < value2:
                    result = 1
                else:
                    result = 0

        return result

