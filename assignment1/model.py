from math import log2
from typing import Protocol

import pandas as pd
import numpy as np


class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, x: pd.DataFrame) -> list:
        ...



class MajorityBaseline(Model):
    def __init__(self):
        pass


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
        '''

        self.most_common_label = pd.Series(y).value_counts().index[0]
        
    

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        return [self.most_common_label] * len(x)
        














class DecisionTree(Model):
    def __init__(self, depth_limit: int = None, ig_criterion: str = 'entropy'):
        '''
        Initialize a new DecisionTree

        Args:
            depth_limit (int): the maximum depth of the learned decision tree. Should be ignored if set to None.
            ig_criterion (str): the information gain criterion to use. Should be one of "entropy" or "collision".
        '''
        
        self.depth_limit = depth_limit
        self.ig_criterion = ig_criterion
        self.feature = None
        self.children = {}
        self.prediction = None
        self.majority = ''
        self.is_trained = False
        self.trained_set = pd.DataFrame()




    # def train(self, x: pd.DataFrame, y: list):
    #     '''
    #     Train a decision tree from a dataset.

    #     Args:
    #         x (pd.DataFrame): a dataframe with the features the tree will be trained from
    #         y (list): a list with the target labels corresponding to each example

    #     Note:
    #         - If you prefer not to use pandas, you can convert a dataframe `df` to a 
    #           list of dictionaries with `df.to_dict(orient='records')`.
    #         - Ignore self.depth_limit if it's set to None
    #         - Use the variable self.ig_criterion to decide whether to calulate information gain 
    #           with entropy or collision entropy
    #     '''

    #     print(f"Training on data: {x.shape}, labels: {len(y)}")
    #     print(f"Training node: depth={self.depth_limit}")
    #     print(f"Stopping condition: {'all labels same' if len(y) == 1 else 'depth limit reached' if self.depth_limit == 0 else 'splitting'}")
    #     print(f"Selected feature: {self.feature}")
    #     print('\n')

    #     majority = pd.Series(y).value_counts().index[0]
    #     self.majority = majority
    #     print(f'majority = {majority}')

    #     if len(y) == 1:
    #         print('All labels are the same. Creating a leaf node.')
    #         self.prediction = majority
    #         return

    #     if self.depth_limit is not None and self.depth_limit == 0:
    #         print('Depth limit reached. Assigning majority class')
    #         self.prediction = majority
    #         return

    #     best_feature = self.select_best_feature(x, y, self.ig_criterion)

    #     if best_feature is None:
    #         print('No valid feature to split on. Assigning majority class.')
    #         self.prediction = majority
    #         return

    #     print(f'Splitting on best feature {best_feature}')
    #     self.feature = best_feature
    #     self.children = {}

    #     split = self.split_dataset(x, y, best_feature)

    #     for subfeature in split.keys():
    #         split_x, split_y = split[subfeature]
    #         print(f'length of split_x = {len(split_x)}. Length of split_y = {split_y}')
    #         # print(f'Creating a child node for feature value: {subfeature}')
    #         child_tree = DecisionTree(depth_limit=None if self.depth_limit is None else self.depth_limit - 1, ig_criterion=self.ig_criterion)
    #         # print('Now am training the child')
    #         child_tree.train(split_x, split_y)
    #         # print('Now am adding child to this trees children')
    #         self.children[subfeature] = child_tree

    #     self.is_trained = True
    #     self.trained_set = x


    def train(self, x: pd.DataFrame, y: list):
        print(f"\n--- Training Node (Depth Left: {self.depth_limit}) ---")
        print(f"Data size: {len(y)}, Features left: {x.columns.tolist()}")
        
        majority = pd.Series(y).value_counts().index[0]
        self.majority = majority
        print(f"Majority class: {majority}")

        # All labels same
        if len(set(y)) == 1:
            self.prediction = y[0]
            print("Stopping: All labels identical.")
            return

        # Depth limit reached
        if self.depth_limit is not None and self.depth_limit <= 0:
            self.prediction = majority
            print("Stopping: Depth limit reached.")
            return

        # No features left to split on
        if len(x.columns) == 0:
            self.prediction = majority
            print("Stopping: No features left.")
            return

        best_feature = self.select_best_feature(x, y, self.ig_criterion)
        print(f"Best feature to split on: {best_feature}")

        if best_feature is None:
            self.prediction = majority
            print("Stopping: No valid feature to split on.")
            return

        self.feature = best_feature
        self.children = {}

        split = self.split_dataset(x, y, best_feature)
        print(f"Splitting on '{best_feature}'. Subgroups: {list(split.keys())}")

        for value, (split_x, split_y) in split.items():
            print(f"\n-> Child node for {best_feature} = {value}")
            print(f"   Subgroup size: {len(split_y)}")
            
            child_tree = DecisionTree(
                depth_limit=None if self.depth_limit is None else self.depth_limit - 1,
                ig_criterion=self.ig_criterion
            )
            child_tree.train(split_x, split_y)
            self.children[value] = child_tree

        self.is_trained = True

        

    
    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''
        
        # print(f'Predicting data on: {x.shape}')
        if self.prediction is not None:
            # print(f'Leaf node prediction: {self.prediction}')
            return [self.prediction] * len(x)

        feature_values = x[self.feature]

        predictions = []

        for _, row in x.iterrows():
            value = row[self.feature]
            
            if value in self.children:
                subtree_predictions = self.children[value].predict(pd.DataFrame([row]))
                predictions.extend(subtree_predictions)
            else:
                valid_children = [child for child in self.children.values() if child.prediction is not None]
                if valid_children:
                    fallback_prediction = max(valid_children, key=lambda c: c.prediction).prediction
                else:
                    fallback_prediction = self.majority
                predictions.append(fallback_prediction)

        return predictions


    




    @staticmethod
    def calculate_entropy(labels: list) -> float: 
        '''
        Calculate the entropy of a set of labels.

        Args: 
            labels (list): A list of target labels
        
        Returns: 
            float: The entropy of the labels        
        '''

        if type(labels) != list: 
            raise ValueError('Input not a valid list')
        if len(labels) == 0:
            return 0
        temp_sum = 0

        for p_i in pd.Series(labels).value_counts():
            temp_sum = temp_sum + p_i / len(labels) * log2(p_i / len(labels))
                
        return -temp_sum

        # for p_i in pd.Series(labels).value_counts():
        #     temp_sum = temp_sum + (p_i / len(labels)) ** 2
                
        # return 1-temp_sum
    





    @staticmethod
    def calculate_collision_entropy(labels: list) -> float: 
        '''
        Calculate the collision entropy of a set of labels

        Args: 
            labels (list): A list of target labels

        Returns: 
            float: The collision entropy of the labels        
        '''
        if type(labels) != list: 
            raise ValueError('Input not a valid list')
        if len(labels) == 0:
            return 0
        
        temp_sum = 0
        for p_i in pd.Series(labels).value_counts():
            temp_sum = temp_sum + (p_i / len(labels))**2
        
        return -log2(temp_sum)
    
    




    def information_gain(self, x: pd.DataFrame, y: list, feature: str, ig_criterion: str) -> float: 
        '''
        Computes the information gain of a splitting on a single given feature. 

        Args: 
            x (pd.DataFrame): A DataFrame of features
            y (list): A list of target labels
            feature (str): the feature over which to calculate information gain for
            ig_criterion (str): The criterion for entropy calculation (either 'entropy' or 'collision')

        Returns: 
            float: The information gain for a single feature
        
        '''

        if ig_criterion not in ['entropy', 'collision']:
            raise ValueError(f"Invalid information gain criterion: {ig_criterion}. Must be 'entropy' or 'collision'")

        split = self.split_dataset(x, y, feature)
        total_size = len(y)

        if ig_criterion == 'entropy':
            weighted_entropy = 0
            for subfeature in split.keys():
                _, split_y = split[subfeature]
                weighted_entropy = weighted_entropy + (len(split_y) / total_size) * self.calculate_entropy(split_y)
            information_gain = self.calculate_entropy(y) - weighted_entropy

        else:
            weighted_collision_entropy = 0
            for subfeature in split.keys():
                _, split_y = split[subfeature]
                weighted_collision_entropy = weighted_collision_entropy + (len(split_y) / total_size) * self.calculate_collision_entropy(split_y)
            information_gain = self.calculate_collision_entropy(y) - weighted_collision_entropy

        return information_gain





    @staticmethod
    def split_dataset(x: pd.DataFrame, y: list, feature: str) -> dict: 
        '''
        Split the dataset based on a given feature.

        Args: 
            x (pd.DataFrame): A dataframe of features
            y (list): A list of target labels
            feature (str): The feature to split on

        Returns: 
            dict: A dictionary where keys are feature values and values are tuples (subset_x, subset_y).    
        '''

        if feature not in x.columns:
            raise KeyError(f'{feature} is not a feature of the DataFrame')
        
        split = {}
        for value in x[feature].unique():
            # print(f'Value = {value}')
            split_x = x[x[feature] == value].drop(columns=[feature])
            # print(f'Split_x = {split_x}')
            subset_indices = x[x[feature] == value].index.tolist()
            split_y = [y[i] for i in range(len(y)) if x.index[i] in subset_indices]
            # print(f'Split_y = {split_y}')
            split[value] = (split_x, split_y)
            
        return split
    


    # def select_best_feature(self, x: pd.DataFrame, y: list, ig_criterion: str) -> str:
    #     '''
    #     Returns the feature for which the information gain is maximal to split on

    #     Args: 
    #         x (pd.DataFrame): A DataFrame of features
    #         y (list): A list of target labels. 
    #         ig_criterion (str): The criteron for splitting, either 'entropy' or 'collision'

    #     Returns: 
    #         str: The name of the best feature for splitting
    #     '''

    #     if ig_criterion not in ['entropy', 'collision']:
    #         raise ValueError(f"Invalid information gain criterion: {ig_criterion}. Must be 'entropy' or 'collision'")
    #     if len(x) != len(y): 
    #         raise IndexError(f'Length of x: {len(x)} =/= length of y: {len(y)}')


    #     best_feature = None
    #     best_information_gain = float('-inf')

    #     for feature in x.columns:
    #         if x[feature].nunique() <= 1:
    #             continue

    #     for feature in x.columns: 
    #         information_gain = self.information_gain(x, y, feature, ig_criterion)
    #         if information_gain > best_information_gain: 
    #             best_information_gain = information_gain
    #             best_feature = feature
        
    #     return best_feature

    def select_best_feature(self, x: pd.DataFrame, y: list, ig_criterion: str) -> str:
        if len(x.columns) == 0:
            return None

        # Randomly select a subset of features (e.g., sqrt(n_features))
        n_features = len(x.columns)
        n_considered = int(np.sqrt(n_features))  # Or a fixed number like 20
        features_to_consider = np.random.choice(x.columns, size=n_considered, replace=False)

        best_feature = None
        best_ig = -float('inf')

        for feature in features_to_consider:  # Only loop over the subset
            if x[feature].nunique() <= 1:
                continue  # Skip constant features

            ig = self.information_gain(x, y, feature, ig_criterion)
            print(f"IG for {feature}: {ig:.4f}")

            if ig > best_ig:
                best_ig = ig
                best_feature = feature

        print(f"Selected best feature: {best_feature} (IG: {best_ig:.4f})")
        return best_feature if best_ig > 0.0 else None  # Stop if no IG improvement
                
                
        
        






