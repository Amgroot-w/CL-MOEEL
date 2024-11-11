import random
from typing import List, TypeVar

from core.operator import Selection
from core.util.comparator import DominanceComparator, SlackComparator, SlackComparator_v1, SlackComparator_v2

S = TypeVar('S')

class RouletteWheelSelection(Selection):
    """ 轮盘赌选择 """

    def __init__(self):
        super(RouletteWheelSelection).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')
            
        # 轮盘赌：采用solution的第一维目标函数作为个体被选择的概率！
        # 即：第一维目标函数越大，该solution被选择到的概率越高！（不适用本问题，因此做如下修改：）
        # 由于在我们的问题中，第一维目标函数是模型的均方根误差，它越小，个体被选择的概率应该越大才对，
        # 因此这里把第一维目标函数取到数，然后再用这个倒数值作为轮盘赌上的数值来进行选择！
        maximum = sum([(1.0 / solution.objectives[0]) for solution in front])
        rand = random.uniform(0.0, maximum)
        value = 0.0
        for solution in front:
            value += 1.0 / solution.objectives[0]
            if value > rand:
                return solution

        return None

    def get_number_of_parents(self) -> int:
        return -1  # 返回-1，表示该选择算子操作的父代个体数为可变的（等于种群大小）

    def get_number_of_children(self) -> int:
        return 1  # 一次选择产生一个子代解

    def get_name(self) -> str:
        return 'Roulette wheel selection'


class BestSolutionSelection(Selection):
    """ 最优解/非支配解选择 """

    def __init__(self):
        super(BestSolutionSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        result = front[0]

        for solution in front[1:]:
            comparator = DominanceComparator()
            if comparator.compare(solution, result) < 0:
                result = solution

        return result

    def get_number_of_parents(self) -> int:
        return -1  # 返回-1，表示该选择算子操作的父代个体数为可变的（等于种群大小）

    def get_number_of_children(self) -> int:
        return 1  # 一次选择产生一个子代解

    def get_name(self) -> str:
        return 'Best solution selection'


class NaryRandomSolutionSelection(Selection):
    """ 随机选择（无放回抽样，一次选择返回所有个体） """

    def __init__(self, number_of_solutions_to_be_returned: int = 1):
        super(NaryRandomSolutionSelection, self).__init__()
        if number_of_solutions_to_be_returned < 0:
            raise Exception('The number of solutions to be returned must be positive integer')
        self.number_of_solutions_to_be_returned = number_of_solutions_to_be_returned

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        if len(front) == 0:
            raise Exception('The front is empty')
        if len(front) < self.number_of_solutions_to_be_returned:
            raise Exception('The front contains less elements than required')

        # random_search sampling without replacement
        return random.sample(front, self.number_of_solutions_to_be_returned)  # 此处返回的是一个列表！

    def get_number_of_parents(self) -> int:
        return -1  # 返回-1，表示该选择算子操作的父代个体数为可变的（等于种群大小）

    def get_number_of_children(self) -> int:
        return self.number_of_solutions_to_be_returned  # 一次选择产生规定的个体数

    def get_name(self) -> str:
        return 'Nary random_search solution selection'


class RandomSolutionSelection(Selection):
    """ 随机选择（有放回抽样，一次选择返回1个个体） """

    def __init__(self):
        super(RandomSolutionSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        return random.choice(front)

    def get_number_of_parents(self) -> int:
        return -1  # 返回-1，表示该选择算子操作的父代个体数为可变的（等于种群大小）

    def get_number_of_children(self) -> int:
        return 1  # 一次选择产生规定的个体数

    def get_name(self) -> str:
        return 'Random solution selection'


class BinaryTournamentSelection(Selection):
    """ 二元锦标赛选择(BTS) """

    def __init__(self):
        super(BinaryTournamentSelection, self).__init__()
        self.comparator = DominanceComparator()  # 支配关系比较器

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')

        elif len(front) == 0:
            raise Exception('The front is empty')

        if len(front) == 1:
            result = front[0]  # 若front中只有一个个体，则直接返回该个体

        else:
            # 先无放回地随机抽取2个解
            i, j = random.sample(range(0, len(front)), 2)
            solution1, solution2 = front[i], front[j]
            # 调用比较器获取两个解的比较结果
            flag = self.comparator.compare(solution1, solution2)
            # 返回较优解，若两个解同等优秀则随机返回
            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]

        return result

    def get_number_of_parents(self) -> int:
        return -1  # 返回-1，表示该选择算子操作的父代个体数为可变的（等于种群大小）

    def get_number_of_children(self) -> int:
        return 1  # 一次选择产生一个子代解

    def get_name(self):
        return 'Binary tournament selection'


class SlackBinaryTournamentSelection(Selection):
    """ 松弛二元锦标赛选择(SlackBTS) """

    def __init__(self):
        super(SlackBinaryTournamentSelection, self).__init__()
        self.comparator = SlackComparator()  # 松弛比较器

    def execute(self, front: List[S], prefer_obj: str = 'RMSE') -> S:
        if front is None:
            raise Exception('The front is null')

        elif len(front) == 0:
            raise Exception('The front is empty')

        if len(front) == 1:
            result = front[0]  # 若front中只有一个个体，则直接返回该个体

        else:
            # 先无放回地随机抽取2个解
            i, j = random.sample(range(0, len(front)), 2)
            solution1, solution2 = front[i], front[j]
            # 调用比较器获取两个解的比较结果
            flag = self.comparator.compare(solution1, solution2, prefer_obj)
            # 返回较优解，若两个解同等优秀则随机返回
            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]

        return result

    def get_number_of_parents(self) -> int:
        return -1  # 返回-1，表示该选择算子操作的父代个体数为可变的（等于种群大小）

    def get_number_of_children(self) -> int:
        return 1  # 一次选择产生一个子代解

    def get_name(self):
        return 'Slack binary tournament selection'


class SlackBinaryTournamentSelection_v1(SlackBinaryTournamentSelection):
    """ 松弛二元锦标赛选择-改进版本1 (SlackBTS_v1) """

    def __init__(self):
        super(SlackBinaryTournamentSelection_v1, self).__init__()
        self.comparator = SlackComparator_v1()  # 更改选择算子的比较器：松弛比较器v1

    def get_name(self):
        return 'Slack binary tournament selection v1'


class SlackBinaryTournamentSelection_v2(SlackBinaryTournamentSelection):
    """ 松弛二元锦标赛选择-改进版本2 (SlackBTS_v2) """

    def __init__(self):
        super(SlackBinaryTournamentSelection_v2, self).__init__()
        self.comparator = SlackComparator_v2()  # 更改选择算子的比较器：松弛比较器v2

    def get_name(self):
        return 'Slack binary tournament selection v2'

