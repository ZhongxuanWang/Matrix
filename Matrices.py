"""
Matrix class written by Daniel Wang encapsulating most of the aspects of matrices. Not like Numpy,
this provides more functions and more user-friendly.
"""


# import numpy as np
# from numpy import array as arr


def arr(arg):
    # For now the numpy array is not used, but it doesn't mean it will never be used.
    return arg


# TODO: In version2.1, optimize the computation

class Matrix:
    __version__ = '2.0'
    __author__ = 'Daniel Wang'

    example_2d_matrix = arr(
        [[4, 1],
         [3, 2]]
    )
    example_3d_matrix = arr(
        [[1, 0, 0],
         [5, 1, 0],
         [0, -3, 1]]
    )
    example_4d_matrix = arr(
        [[2, -1, 0, 0],
         [-1, 2, -1, 0],
         [0, -2, 2, -1],
         [0, 0, -1, 2]]
    )

    def __init__(self, mat, shape=None):
        if mat is not None:
            self.mat = arr(mat)
        else:
            assert shape is not None, 'When the matrix is not given, the shape cant be empty!'
            self.mat = self.identity(shape)

    def get_mat(self):
        return self.__mat

    def set_mat(self, other):
        assert len(other) > 0, 'The matrix cant be empty'
        self.nrows = self.ndim = len(other)
        self.ncols = len(other[0])
        self.shape = (self.nrows, self.ncols)

        self.__mat = arr(other)

    mat = property(get_mat, set_mat)

    def __str__(self):
        return "Matrix" + str(self.mat)

    def __mul__(self, other):
        return self.__internal_dot(other)

    def __add__(self, other):
        return self.__internal_add(other)

    def __pow__(self, power, modulo=None):
        m = self.mat.copy()
        for a in range(power):
            m = self.__internal_dot(m)
        return m

    def __abs__(self):
        return self.det()

    # Make the object from this class subscriptable
    def __getitem__(self, item):
        return self.mat[item]

    def __len__(self):
        return self.nrows

    def dot(self, nm, inplace=False):
        result = self.__internal_dot(nm)
        if inplace:
            self.mat = result
        return result

    def add(self, nm, inplace=False):
        result = self.__internal_add(nm)
        if inplace:
            self.mat = result
        return result

    def inv(self):
        det = self.det()
        assert det is not 0, 'only matrix with none-zero determinant have its inverse matrix!'
        row = len(self.mat)
        col = len(self.mat[0])
        # Make sure it's a square
        assert row == col
        return self.__internal_inv(det)

    def det(self, limit=None):
        row = len(self.mat)
        col = len(self.mat[0])
        # Make sure it's a square
        assert row == col, 'only squared matrix have determinant!'
        if limit is None:
            limit = 1000000
        # Edge cases
        if 997 < row < limit:
            import sys
            sys.setrecursionlimit(row + 2)
        elif row > limit:
            raise Exception("Row/Col number exceeded limit.")
        return self.__internal_det(self.mat)

    '''
    Transpose = Interhange row and column
    Get the transpose matrix of this matrix
    '''

    def transpose(self, other=None, inplace=False):
        nm = self.__internal_transpose(other)
        if inplace:
            self.mat = nm
        return nm

    '''
    Get the adjoint matrix of this matrix
    '''

    def adj(self, inplace=False):
        adj = self.__internal_adj()
        if inplace:
            self.mat = adj
        return adj

    '''
    Get the cofactor matrix of this matrix
    '''

    def cof(self, inplace=False):
        cof = self.__internal_cof()
        if inplace:
            self.mat = cof
        return cof

    def argmax(self):
        return max(self.mat)

    def argmin(self):
        return min(self.mat)

    # This means a private method
    def __internal_dot(self, nm):
        if isinstance(nm, (int, float, bool)):
            nm = self.mat.copy()
            return self.mat * nm
        else:
            assert len(self.mat[0]) == len(nm)

            matcher = len(self.mat[0])

            nrow_of_mat, ncol_of_new_mat = len(self.mat), len(nm[0])

            new_mat = self.fill((nrow_of_mat, ncol_of_new_mat))

            for row in range(nrow_of_mat):
                for col in range(ncol_of_new_mat):
                    nsum = 0
                    for match in range(matcher):
                        nsum += nm[match][col] * self.mat[row][match]
                    new_mat[row][col] = nsum
            return new_mat

    def __internal_add(self, nm):
        if isinstance(nm, (int, float, bool, arr)):
            return self.mat + nm
        else:
            return self.mat + arr(nm)
            # assert len(self.mat) == len(nm) and len(self.mat[0]) == len(nm[0])
            # for row in range(len(nm)):
            #     for col in range(len(nm[0])):
            #         nm[row][col] += self.mat[row][col]
            # return nm

    def __internal_inv(self, det):
        return self.adj() / det

    """
    Precondition: 
    1. Squre matrix
    
    Postcondition:
    mat object is not changed    
    """

    def __internal_det(self, nm):
        number_sum = 0
        if len(nm) == 1:
            return nm[0]
        elif len(nm) == 2:
            return nm[0][0] * nm[1][1] - nm[0][1] * nm[1][0]
        else:
            for col in range(len(nm[0])):
                if nm[0][col] == 0:
                    continue
                # number_sum += (-1) ** (2 + col) * nm[0][col] * self.__internal_det(
                #     np.delete(
                #         np.delete(nm, 0, 0), col, 1)
                # )
                number_sum += (-1) ** (2 + col) * nm[0][col] * self.__internal_det(
                    Matrix.delete(nm, (0, col))
                )
        return number_sum

    # def __construct_nm(self, nm, col):
    #     changed_matrix = Matrix.zero(len(nm) - 1, len(nm[0]) - 1)
    #     for r in range(1, len(nm)):
    #         for c in range(0, len(nm[0])):
    #             if c != col:
    #                 changed_matrix = nm[r][c]
    #     return changed_matrix

    def __internal_transpose(self, mat):
        if mat is None:
            mat = self.mat.copy()
        new_mat = Matrix.zeros((len(mat[0]), len(mat)))
        for a in range(len(mat[0])):
            new_mat[a] = mat[:, a]
        return new_mat

    def __internal_adj(self):
        return self.__internal_transpose(self.__internal_cof())

    def __internal_cof(self):
        nrows, ncols = len(self.mat), len(self.mat[0])
        nm = Matrix.fill((nrows, ncols), 0)
        if nrows == 1:
            nm[0] = self.mat[0]
        elif nrows == 2:
            nm[0, 0] = self.mat[1, 1]
            nm[0, 1] = -self.mat[1, 0]
            nm[1, 0] = -self.mat[0, 1]
            nm[1, 1] = self.mat[0, 0]
        else:
            for row in range(nrows):
                for col in range(ncols):
                    nm[row][col] = (-1) ** (2 + row + col) * \
                                   self.__internal_det(
                                        Matrix.delete(nm, (0, col))
                                   )
        return nm

    @staticmethod
    def delete(mat, loc):
        row, col = loc
        nrow, ncol = len(mat) - 1, len(mat[0]) - 1
        nm = Matrix.fill((nrow, ncol))
        tr = 0
        tc = 0
        for i, r in enumerate(mat):
            for j, c in enumerate(r):
                if i == row:
                    tr -= 1
                    break
                if j == col:
                    continue
                nm[tr][tc] = c
                tc += 1
            tc = 0
            tr += 1
        return nm

    # @staticmethod
    # def reshape(mat, newshape):
    #     ncols, nrows = newshape
    #     nm = Matrix.fill(newshape)

    @staticmethod
    def identity(shape):
        row, col = shape
        assert row > 0 and col > 0
        a = Matrix.zeros(shape)
        for i in range(row):
            for j in range(col):
                if i == j:
                    a[i][j] = 1
        return a

    @staticmethod
    def fill(shape, content=None):
        row, col = shape
        assert row > 0 and col > 0
        return arr([[content for _ in range(col)] for _ in range(row)])

    @staticmethod
    def zeros(shape):
        return Matrix.fill(shape, content=0)

    @staticmethod
    def ones(shape):
        return Matrix.fill(shape, content=1)

    @staticmethod
    def flatten(mat):
        return [item for sublist in mat for item in sublist]

    # @staticmethod
    # def rand(shape):
    #     nrows, ncols = shape
    #     assert nrows > 0 and ncols > 0, 'The number of rows and columns must be bigger than 0!'
    #     return Matrix(mat=np.random.randn(nrows, ncols))
    #
    # @staticmethod
    # def randint(min, max, shape):
    #     assert shape[0] > 0 and shape[1] > 0, 'The number of rows and columns must be bigger than 0!'
    #     return Matrix(mat=np.random.randint(min, max, size=shape))
