import numpy as np
from random import shuffle
from past.builtins import xrange

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
  j = 0
  for i in range(num_train):
      scores = X[i].dot(W)
      j = j + scores
      correct_class_score = scores[y[i]]
      for j in range(num_classes):
          if j == y[i]:
              continue
          margin = scores[j] - correct_class_score + 1  # note delta = 1
          if margin > 0:
              loss += margin
              dW[:, j] += X[i]
              dW[:, y[i]] -= X[i]
              # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X, W)
  correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
  margine = np.maximum(0, scores - correct_class_score + 1)
  margine[np.arange(num_train), y] = 0
  loss = np.sum(margine) / num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask = margine
  mask[margine > 0] = 1
  sum_classes = np.sum(mask, axis=1)
  mask[np.arange(num_train), y] = -sum_classes.T
  dW = np.dot(X.T, mask)
  dW /= num_train

  return loss, dW
