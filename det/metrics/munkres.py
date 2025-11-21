#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :munkres.py
@Author :CodeCat
@Date   :2025/11/21 11:07
"""
"""
This code is based on https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/eval_kitti_track/munkres.py
"""

import sys
import copy


def make_cost_matrix(profit_matrix, inversion_function):
    """
    通过调用'inversion_function'来反转每个值，从效益矩阵创建代价矩阵
    反转函数必须接受一个数值参数（任何类型）并返回另一个数值参数，
    该参数被假定为原始效益的代价逆值

    这是一个静态方法。像这样调用它：

    .. python::

        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)

    例如：

    .. python::

        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxint - x)

    :Parameters:
        profit_matrix : list of lists
            要从效益转换为代价矩阵的矩阵

        inversion_function : function
            用于反转效益矩阵中每个条目的函数

    :rtype: list of lists
    :return: 转换后的矩阵
    """
    cost_matrix = []
    for row in profit_matrix:
        cost_matrix.append([inversion_function(value) for value in row])
    return cost_matrix


class Munkres:
    """
    计算经典分配问题的Munkres解。
    """

    def __init__(self):
        """创建一个新实例"""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def make_cost_matrix(profit_matrix, inversion_function):
        """
        **已弃用**

        请使用模块函数 ``make_cost_matrix()``。
        """
        import munkres
        return munkres.make_cost_matrix(profit_matrix, inversion_function)

    make_cost_matrix = staticmethod(make_cost_matrix)

    @staticmethod
    def pad_matrix(matrix, pad_value=0):
        """
        填充一个可能不是方阵的矩阵以使其成为方阵。

        :Parameters:
            matrix : list of lists
                要填充的矩阵

            pad_value : int
                用于填充矩阵的值

        :rtype: list of lists
        :return: 一个新的，可能被填充的矩阵
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                # 行太短。填充它。
                new_row += [pad_value] * (total_rows - row_len)
            new_matrix += [new_row]

        while len(new_matrix) < total_rows:
            new_matrix += [[pad_value] * total_rows]

        return new_matrix

    def compute(self, cost_matrix):
        """
        计算数据库中行和列之间的最低成本配对的索引。
        返回一个可用于遍历矩阵的(row, column)元组列表。

        :Parameters:
            cost_matrix : list of lists
                代价矩阵。如果此代价矩阵不是方阵，
                它将通过调用 ``pad_matrix()`` 填充为零。
                （此方法不修改调用者的矩阵。它在矩阵副本上操作。）

                **警告**: 此代码处理方阵和矩形矩阵。
                它不处理不规则矩阵。

        :rtype: list
        :return: 描述矩阵中最低成本路径的 ``(row, column)`` 元组列表

        """
        self.C = self.pad_matrix(cost_matrix)
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0])
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_matrix(self.n * 2, 0)
        self.marked = self.__make_matrix(self.n, 0)

        done = False
        step = 1

        steps = {
            1: self.__step1,
            2: self.__step2,
            3: self.__step3,
            4: self.__step4,
            5: self.__step5,
            6: self.__step6
        }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # 查找被标记的列
        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.marked[i][j] == 1:
                    results += [(i, j)]

        return results

    @staticmethod
    def __copy_matrix(matrix):
        """返回所提供矩阵的精确副本"""
        return copy.deepcopy(matrix)

    @staticmethod
    def __make_matrix(n, val):
        """创建一个 *n*x*n* 矩阵，用特定值填充它。"""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self):
        """
        对于矩阵的每一行，找到最小元素并从该行的每个元素中减去它。
        转到步骤2。
        """
        C = self.C
        n = self.n
        for i in range(n):
            minval = min(self.C[i])
            # 找到此行的最小值并从行中的每个元素中减去该最小值。
            for j in range(n):
                self.C[i][j] -= minval

        return 2

    def __step2(self):
        """
        在结果矩阵中找到一个零(Z)。
        如果其行或列中没有被标记的零，则标记Z。
        对矩阵中的每个元素重复此操作。
        转到步骤3。
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.C[i][j] == 0) and \
                   (not self.col_covered[j]) and \
                   (not self.row_covered[i]):
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True

        self.__clear_covers()
        return 3

    def __step3(self):
        """
        覆盖包含标记零的每一列。
        如果K列被覆盖，则标记的零描述了一组完整的唯一分配。
        在这种情况下，转到完成，否则，转到步骤4。
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == 1:
                    self.col_covered[j] = True
                    count += 1

        if count >= n:
            step = 7  # 完成
        else:
            step = 4

        return step

    def __step4(self):
        """
        找到一个未被覆盖的零并将其标记为素数。
        如果包含此素数零的行中没有被标记的零，则转到步骤5。
        否则，覆盖此行并取消覆盖包含被标记零的列。
        继续以这种方式进行，直到没有未被覆盖的零为止。
        保存最小的未覆盖值并转到步骤6。
        """
        step = 0
        done = False
        row = -1
        col = -1
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero()
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self):
        """
        构造一系列交替的素数零和标记零，如下所示。
        让Z0表示在步骤4中找到的未覆盖的素数零。
        让Z1表示Z0列中的标记零(如果有的话)。
        让Z2表示Z1行中的素数零(总是会有一个)。
        继续直到系列在没有标记零的列的素数零处终止。
        取消标记系列中的每个标记零，标记系列中的每个素数零，
        擦除所有素数并取消覆盖矩阵中的每一行。
        返回步骤3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count - 1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count - 1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self):
        """
        将在步骤4中找到的值添加到每个被覆盖行的每个元素，
        并从每个未被覆盖列的每个元素中减去它。
        返回步骤4，不改变任何标记、素数或覆盖线。
        """
        minval = self.__find_smallest()
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i][j] += minval
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
        return 4

    def __find_smallest(self):
        """在矩阵中找到最小的未覆盖值。"""
        minval = 2e9  # sys.maxint
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval

    def __find_a_zero(self):
        """找到第一个未覆盖的值为0的元素"""
        row = -1
        col = -1
        i = 0
        n = self.n
        done = False

        while not done:
            j = 0
            while True:
                if (self.C[i][j] == 0) and \
                   (not self.row_covered[i]) and \
                   (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j += 1
                if j >= n:
                    break
            i += 1
            if i >= n:
                done = True

        return (row, col)

    def __find_star_in_row(self, row):
        """
        找到指定行中的第一个标记元素。
        返回列索引，如果没有找到标记元素，则返回-1。
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col):
        """
        找到指定列中的第一个标记元素。
        返回行索引，如果没有找到标记元素，则返回-1。
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row):
        """
        找到指定行中的第一个素数元素。
        返回列索引，如果没有找到标记元素，则返回-1。
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self, path, count):
        for i in range(count + 1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def __clear_covers(self):
        """清除所有被覆盖的矩阵单元"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self):
        """擦除所有素数标记"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0