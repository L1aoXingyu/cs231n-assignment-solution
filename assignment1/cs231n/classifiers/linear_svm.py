import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # 如果 margin 大于 0，计算梯度
                dW[:, j] += X[i]
                dW[:, y[i]] += -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # 对梯度除以 num_train 进行平均
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # 对正则项求梯度
    dW += 2 * reg * W
    #############################################################################
    # TODO:
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]  # 得到样本的数目
    scores = np.dot(X, W)  # 计算所有的得分
    y_score = scores[np.arange(num_train), y].reshape((-1, 1))  # 得到每个样本对应label的得分
    mask = (scores - y_score + 1) > 0  # 有效的score下标
    scores = (scores - y_score + 1) * mask  # 有效的得分
    loss = (np.sum(scores) - num_train * 1) / num_train  # 去掉每个样本多加的对应label得分，然后平均
    loss += reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # dw = x.T * dl/ds
    ds = np.ones_like(scores)  # 初始化ds
    ds *= mask  # 有效的score梯度为1，无效的为0
    ds[np.arange(num_train), y] = -1 * (np.sum(mask, axis=1) - 1)  # 每个样本对应label的梯度计算了(有效的score次)，取负号
    dW = np.dot(X.T, ds) / num_train   # 平均
    dW += 2 * reg * W  # 加上正则项的梯度
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
