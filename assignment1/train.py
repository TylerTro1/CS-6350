import argparse
import pandas as pd
import numpy as np

from data import load_data
from model import DecisionTree, MajorityBaseline, Model


def train(model: Model, x: pd.DataFrame, y: list):
    '''
    Learn a model from training data.

    Args:
        model (Model): an instantiated MajorityBaseline or DecisionTree model
        x (pd.DataFrame): a dataframe with the features the tree will be trained from
        y (list): a list with the target labels corresponding to each example
    '''
    
    model.train(x, y)
    

def evaluate(model: Model, x: pd.DataFrame, y: list) -> float:
    '''
    Evaluate a trained model against a dataset

    Args:
        model (Model): an instance of a MajorityBaseline model or a DecisionTree model
        x (pd.DataFrame): a dataframe with the features the tree will be predicted from
        y (list): a list with the target labels corresponding to each example

    Returns:
        float: the accuracy of the decision tree's predictions on x, when compared to y
    '''
       
    return calculate_accuracy(y, model.predict(x))




def calculate_accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.
    Should be a float between 0 and 1.

    Args:
        labels (list): the ground-truth labels from the data
        predictions (list): the predicted labels from the model

    Returns:
        float: the accuracy of the predictions, when compared to the ground-truth labels
    '''


    if len(labels) != len(predictions):
        raise ValueError(f'Length of labels list: {len(labels)} is not equal to the length of predictions list: {len(predictions)}')
    # if len(labels) == 0:
        # raise ValueError('Labels list is empty')
    
    return np.mean(np.array(labels) == np.array(predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model_type', '-m', type=str, choices=['majority_baseline', 'decision_tree'], 
        help='Which model type to train')
    parser.add_argument('--depth_limit', '-d', type=int, default=None, 
        help='The maximum depth of a DecisionTree. Ignored if model_type is not "decision_tree".')
    parser.add_argument('--ig_criterion', '-i', type=str, choices=['entropy', 'collision'], default='entropy',
        help='Which information gain variant to use. Ignored if model_type is not "decision_tree".')
    args = parser.parse_args()


    # load data
    data_dict = load_data()

    train_df = data_dict['train']
    train_x = train_df.drop('label', axis=1)
    train_y = train_df['label'].tolist()

    test_df = data_dict['test']
    test_x = test_df.drop('label', axis=1)
    test_y = test_df['label'].tolist()


    # initialize the model
    if args.model_type == 'majority_baseline':
        model = MajorityBaseline()
    elif args.model_type == 'decision_tree':
        model = DecisionTree(depth_limit=args.depth_limit, ig_criterion=args.ig_criterion)
    else:
        raise ValueError(
            '--model_type must be one of "majority_baseline" or "decision_tree". ' +
            f'Received "{args.model_type}". ' +
            '\nRun `python train.py --help` for additional guidance.')


    # train the model
    train(model=model, x=train_x, y=train_y)


    # evaluate model on train and test data
    train_accuracy = evaluate(model=model, x=train_x, y=train_y)
    print(f'train accuracy: {train_accuracy:.3f}')
    test_accuracy = evaluate(model=model, x=test_x, y=test_y)
    print(f'test accuracy: {test_accuracy:.3f}')
