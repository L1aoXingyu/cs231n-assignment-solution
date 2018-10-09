

class Matrix(object):
    def __init__(self, x):
        self.element = x
        self.h = len(x)
        self.w = len(x[0])

    def dot(self, y):
        e = y.element
        y_h = len(e)
        y_w = len(e[0])
        assert self.w == y_h
        results = []
        for i in range(self.h):
            result = []
            for j in range(y_w):
                r = 0
                for k in range(self.w):
                    r += self.element[i][k] * e[k][j]
                result.append(r)
            results.append(result)
        return Matrix(results)

    def __str__(self):
        vis = ''
        for i in self.element:
            vis += (str(i))
            vis += '\n'
        return vis

    def __repr__(self):
        return 'This is a Matrix class'


if __name__ == '__main__':
    A = Matrix([[2, 3], [4, 5], [3, 4]])
    B = Matrix([[2, 3, 3], [3, 5, 9]])
    print(A.dot(B))

    print('numpy test')
    import numpy as np
    a = np.array([[2, 3], [4, 5], [3, 4]])
    b = np.array([[2, 3, 3], [3, 5, 9]])
    print(np.dot(a, b))

    # from IPython import embed
    # embed()
