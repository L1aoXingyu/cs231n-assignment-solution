import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                diff = X[i] - self.X_train[j]  # (x1-y1, x2-y2, ... xd-yd)
                diff_2 = diff ** 2  # ((x1-y1)^2, (x2-y2)^2, ... (xd-yd)^2)
                d = np.sqrt(np.sum(diff_2))  # sqrt((x1-y1)^2 + (x2-y2)^2 + ... (xd-yd)^2)
                dists[i, j] = d
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            diff = self.X_train - X[i]  # 回忆一下什么是broadcast
            dist = np.sum(diff ** 2, axis=1)  #沿着行求和，即 (x1-y1)^2 + (x2-y2)^2 + ...
            dists[i, :] = np.sqrt(dist)  # 对平方和开根号
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # (x1 - y1)^2 + (x2 - y2)^2 = (x1^2 + x2^2) + (y1^2 + y2^2) - 2*(x1*y1 + x2*y2)
        train_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True)  # (m, 1), 注意 keepdims 的含义
        train_sq = np.broadcast_to(train_sq, shape=(num_train, num_test)).T  # (n, m), 注意转置
        test_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
        test_sq = np.broadcast_to(test_sq, shape=(num_test, num_train))  # (n, m)
        cross = np.dot(X, self.X_train.T)  # (n, m)
        dists = np.sqrt(train_sq + test_sq - 2 * cross)  # 开根号
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            test_dist = dists[i]  # 每个test样本和train set里面的距离
            sort_dist = np.argsort(test_dist)  # 得到排序下标，从小到大，即第一个下标表示最近的训练样本
            valid_idx = sort_dist[:k]  # 只取前k个
            closest_y = self.y_train[valid_idx]  # 获得前k个的label, 比如是 (1, 2, 2, 3)
            #########################################################################
            # TODO:
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # 查看 np.unique 函数
            y_unique, y_count = np.unique(closest_y, return_counts=True)
            # 第一个是去掉closest_y中的重复元素，第二个是剩下的元素出现的次数
            common_idx = np.argmax(y_count)  # 出现次数最多的位置
            y_pred[i] = y_unique[common_idx]  # 取最大出现次数的label作为预测
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_pred
