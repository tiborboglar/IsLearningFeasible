import numpy as np
from util import randomize_in_place


def linear_regression_prediction(X, w):
    """
    Calculando a predição da regressão linear.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: pesos
    :type w: np.array(shape=(d, 1))
    :return: predição
    :rtype: np.array(shape=(N, 1))
    """

    return X.dot(w)


def standardize(X):
    """
    'Normalização' Z-Score

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: array 'normalizado'
    :rtype: np.ndarray(shape=(N, d))
    """

    X_temp = (X.T - np.mean(X))/np.std(X)
    X_out = X_temp.T

    return X_out

def compute_cost(X, y, w):
    """
    Computa a função custo, minimizando o Mean Squared Error.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: targets da regressão
    :type y: np.ndarray(shape=(N, 1))
    :param w: pesos
    :type w: np.array(shape=(d,))
    :return: custo
    :rtype: float
    """

    J_temp = (1/len(X)) * np.matmul((np.matmul(X,w) - y).T,(np.matmul(X,w) - y))

    J = J_temp[0][0]

    return J


def compute_wgrad(X, y, w):
    """
    Calculando o gradiente J(w) em que w são os pesos

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: targets da regressão
    :type y: np.ndarray(shape=(N, 1))
    :param w: pesos
    :type w: np.array(shape=(d,))
    :return: gradiente
    :rtype: np.array(shape=(d,))
    """

    grad = (2/len(X)) * np.matmul((np.matmul(X,w) - y).T, X).T


    return grad


def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
    Utiliza o gradiente descendente em batches, otimizando o gradiente descendente.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: targets da regressão
    :type y: np.ndarray(shape=(N, 1))
    :param w: pesos
    :type w: np.array(shape=(d,))
    :param learning_rate: taxa de aprendizado
    :type learning_rate: float
    :param num_iters: número de iterações
    :type num_iters: int
    :return: pesos, pesos históricos, custos históricos
    :rtype: np.array(shape=(d,)), list, list
    """

    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]
    
    for i in range (0, num_iters):
        w = w - learning_rate*compute_wgrad(X, y, w)
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X,y,w))

    return w, weights_history, cost_history


def stochastic_gradient_descent(X, y, w, learning_rate, num_iters, batch_size):
    """
    Algoritmo estocástico do gradiente descendente, também otimizado.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: targets da regressão
    :type y: np.ndarray(shape=(N, 1))
    :param w: pesos
    :type w: np.array(shape=(d, 1))
    :param learning_rate: taxa de aprendizado
    :type learning_rate: float
    :param num_iters: número de iterações
    :type num_iters: int
    :param batch_size: tamanho do minibatch
    :type batch_size: int
    :return: pesos, pesos históricos, custos históricos
    :rtype: np.array(shape=(d, 1)), list, list
    """
    
    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]
    
    for i in range (0, num_iters):
        
        random_num = np.random.randint(X.shape[0], size=batch_size)
        X_temp = X[random_num, :]
        y_temp = y[random_num, :]
        
        w = w - learning_rate*compute_wgrad(X_temp, y_temp, w)
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X_temp,y_temp,w))

    return w, weights_history, cost_history
