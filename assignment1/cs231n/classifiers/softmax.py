import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(X.shape[0]):
        score = np.dot(X[i], W)
        score -= max(score)  # 为了数值稳定性
        score = np.exp(score)  # 取指数
        softmax_sum = np.sum(score)  # 得到分母
        score /= softmax_sum  # 除以分母得到softmax
        # 计算梯度
        for j in range(W.shape[1]):
            if j != y[i]:
                dW[:, j] += score[j] * X[i]
            else:
                dW[:, j] -= (1 - score[j]) * X[i]

        loss -= np.log(score[y[i]])  # 得到交叉熵
    loss /= X.shape[0]  # 平均
    dW /= X.shape[0]  # 平均
    loss += reg * np.sum(W * W)  # 加上正则项
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1, keepdims=True)  # 数值稳定性
    scores = np.exp(scores)
    scores /= np.sum(scores, axis=1, keepdims=True)  # softmax
    ds = np.copy(scores)
    ds[np.arange(X.shape[0]), y] -= 1
    dW = np.dot(X.T, ds)
    loss = scores[np.arange(X.shape[0]), y]
    loss = -np.log(loss).sum()
    loss /= X.shape[0]
    dW /= X.shape[0]
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
