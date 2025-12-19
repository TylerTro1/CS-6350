''' This file defines the model classes that will be used. 
'''

from typing import Protocol, Tuple

import numpy as np
from collections import Counter

np.random.seed(1)



class Model(Protocol):
    def get_hyperparams(self) -> dict:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...














class MajorityBaseline(Model):
    def __init__(self):
        self.majority = None

    def get_hyperparams(self) -> dict:
        return {}

    def train(self, x: np.ndarray, y: np.ndarray):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example

        Hints:
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
        '''

        # self.majority = np.argmax(np.bincount(y + 1)) - 1
        labels, counts = np.unique(y, return_counts=True)
        self.majority = labels[np.argmax(counts)]
        

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        return [self.majority] * x.shape[0]





















class Perceptron(Model):
    def __init__(self, num_features: int, lr: float, decay_lr: bool = False, mu: float = 0):
        '''
        Initialize a new Perceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate (eta). This is also the initial learning rate if decay_lr=True
            decay_lr (bool): whether or not to decay the initial learning rate lr
            mu (float): the margin (mu) that determines the threshold for a mistake. Defaults to 0
        '''

        self.lr = lr
        self.decay_lr = decay_lr
        self.mu = mu
        self.w_t = np.zeros(num_features) + np.random.uniform(-0.01, 0.01, num_features)
        self.b = np.random.uniform(-0.01, 0.01)
        self.t = 0
        self.total_updates = 0

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr, 'decay_lr': self.decay_lr, 'mu': self.mu}

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        for j in range(1, epochs + 1):
            if j > 1:
                x, y = shuffle_data(x, y)
            for i in range(len(x)):
                self.weight_update(x[i], y[i])
            self.t = self.t + 1

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        y = []
        for i in range(len(x)):
            y.append(self.predicted_label(x[i]))
        return y

    def predicted_label(self, x_i: np.ndarray):
        ''' 
        Compute the predicted label using the current weights vector. 

        Args: 
            x_i (np.ndarray): Represents that i_th slice of the x np.ndarray used in the predict method
        '''
        return np.sign(np.dot(self.w_t, x_i) + self.b)

    def weight_update(self, x_i, y_i):
        '''
        Update the weight vector if the prediction is wrong. 

        Args: 
            x_i (np.ndarray): Represents the i_th slice of the x np.ndarray used in the train method
            y_i (np.ndarray): Represents the i_th element of the y np.ndarray used in the train method
        '''

        if y_i * np.dot(self.w_t, x_i) + self.b <= self.mu:
            if self.decay_lr == False:
                self.w_t = self.w_t + self.lr * y_i * x_i
                self.b = self.b + self.lr * y_i
            else:
                self.w_t = self.w_t + (self.lr / (self.t + 1)) * y_i * x_i
                self.b = self.b + (self.lr / (self.t + 1)) * y_i   
            self.total_updates += 1 

    def get_total_updates(self):
        return self.total_updates































class AveragedPerceptron(Model):
    def __init__(self, num_features: int, lr: float):
        '''
        Initialize a new AveragedPerceptron.

        Args:
            num_features (int): The number of features (i.e., dimensions) the model will have.
            lr (float): The learning rate η.
        '''     

        self.lr = lr
        self.w_t = np.zeros(num_features) + np.random.uniform(-0.01, 0.01, num_features)
        self.w_a = np.zeros(num_features)
        self.b = np.random.uniform(-0.01, 0.01)
        self.b_a = 0                                
        self.a = np.zeros(num_features)
        self.count = 1
        self.total_updates = 0
        

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr}

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train the model using the Perceptron algorithm.

        Args:
            x (np.ndarray): A 2-D (num_examples x num_features) array of training examples.
            y (np.ndarray): A 1-D (num_examples) array of labels corresponding to each example.
            epochs (int): Number of epochs to train.

        Hints:
            - Remember to shuffle your data between epochs.
            - `np.matmul()` is useful for matrix multiplication.
        '''
        for j in range(1, epochs + 1):
            if j > 1:
                x, y = shuffle_data(x, y)
            for i in range(len(x)):
                self.average_weight_update(x[i], y[i])
        self.b_a /= self.count # TESTING
        self.a /= self.count # TESTING
    
    def predict(self, x: np.ndarray) -> list:
        '''
        Predict labels for input data.

        Args:
            x (np.ndarray): A 2-D (num_examples x num_features) array of feature vectors.

        Returns:
            np.ndarray: A 1-D array of predicted labels.
        '''
        return np.sign(np.dot(x, self.a))

    def average_predicted_label(self, x_i: np.ndarray):
        '''Compute the predicted label using the averaged weight vector.'''
        return np.sign(np.dot(self.a, x_i))

    def average_weight_update(self, x_i: np.ndarray, y_i: int): 
        '''
        Update the weights if the prediction is incorrect.

        Args:
            x_i (np.ndarray): A single feature vector.
            y_i (int): The corresponding label (+1 or -1).
        '''
        if y_i * np.dot(self.w_t, x_i) <= 0:
            self.w_t += self.lr * y_i * x_i
            self.total_updates += 1
        self.a += self.w_t
        self.b_a += self.b 
        self.count += 1

    def get_total_updates(self): 
        return self.total_updates










































class AggressivePerceptron(Model):
    def __init__(self, num_features: int, mu: float):
        '''
        Initialize a new AggressivePerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            mu (float): the hyperparameter mu
        '''     

        self.mu = mu
        self.w_t = np.zeros(num_features) + np.random.uniform(-0.01, 0.01, num_features)
        self.b = np.random.uniform(-0.01, 0.01)
        self.total_updates = 0

    def get_hyperparams(self) -> dict:
        return {'mu': self.mu}


    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''
        for j in range(1, epochs + 1):
            if j > 1:
                x, y = shuffle_data(x, y)
            for i in range(len(x)):
                self.weight_update(x[i], y[i])


    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in x.
        '''

        return np.sign(np.dot(x, self.w_t))
    
    def predicted_label(self, x_i: np.ndarray):
        '''
        Compute the predicted label using the averaged weight vector.

        Args: 
            x_i (np.ndarray): Represents the i_th element of the feature data
        '''
        return np.sign(np.dot(self.w_t, x_i))

    def weight_update(self, x_i: np.ndarray, y_i: int): 
        '''
        Update the weights if the prediction is incorrect.

        Args:
            x_i (np.ndarray): A single feature vector.
            y_i (int): The corresponding label (+1 or -1)
        '''
        if y_i * np.dot(self.w_t, x_i) + self.b <= self.mu:
            self.w_t += (self.mu - y_i * np.dot(self.w_t, x_i)) / (np.dot(x_i, x_i) + 1) * y_i * x_i
            self.total_updates += 1

    def get_total_updates(self): 
        return self.total_updates










class PerceptronEnsemble(Model):
    def __init__(self, num_features: int, estimator_count: int, lr_values: list, decay_bools: list, mu_values: list):
        self.num_features = num_features
        self.estimator_count = estimator_count
        self.estimators = []
        self.lr_values = lr_values
        self.mu_values = mu_values
        self.decay_bools = decay_bools

    def train(self, X, y, epochs):
        for i in range(self.estimator_count):
            # Train a perceptron
            perceptron = Perceptron(self.num_features, self.lr_values[i], self.decay_bools[i], self.mu_values[i])
            perceptron.train(X, y, epochs)
            self.estimators.append(perceptron)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])

        predictions = ((predictions + 1) // 2).astype(int)  # Now values are 0 or 1

        predictions = predictions.T

        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

        return majority_votes



































PERCEPTRON_VARIANTS = ['simple', 'decay', 'margin', 'averaged', 'aggressive']
MODEL_OPTIONS = ['majority_baseline'] + PERCEPTRON_VARIANTS

def init_perceptron(variant: str, num_features: int, lr: float, mu: float) -> Model:
    '''
    This is a helper function to help you initialize the correct variant of the Perceptron

    Args:
        variant (str): which variant of the perceptron to use. See PERCEPTRON_VARIANTS above for options
        num_features (int): the number of features (i.e. dimensions) the model will have
        lr (float): the learning rate hyperparameter eta. Same as initial learning rate for decay setting
        mu (float): the margin hyperparamter mu. Ignored for variants "simple", "decay", and "averaged"

    Returns
        (Model): the initialized perceptron model
    '''

    assert variant in PERCEPTRON_VARIANTS, f'{variant=} must be one of {PERCEPTRON_VARIANTS}'

    if variant == 'simple':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=False)
    elif variant == 'decay':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True)
    elif variant == 'margin':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True, mu=mu)
    elif variant == 'averaged':
        return AveragedPerceptron(num_features=num_features, lr=lr)
    elif variant == 'aggressive':
        return AggressivePerceptron(num_features=num_features, mu=mu)
    

def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''

    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]