from typing import Protocol, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import DecisionTree
from train import evaluate, train


print('WORKING \n')

train_data = pd.read_csv('train.csv')
train_labels = train_data['label']
train_labels = train_labels.to_list()
train_data = train_data.drop(columns='label')

test_data = pd.read_csv('test.csv')
test_labels = test_data['label']
test_labels = test_labels.to_list()
test_data = test_data.drop(columns='label')

eval_anon_data = pd.read_csv('eval.anon.csv')
eval_anon_labels = eval_anon_data['label']
eval_anon_labels = eval_anon_labels.to_list()
eval_anon_data = eval_anon_data.drop(columns = 'label')


from sklearn.feature_selection import VarianceThreshold

# Drop features with low variance (e.g., 95% same value)
selector = VarianceThreshold(threshold=0.05 * (1 - 0.05))
train_data_reduced = selector.fit_transform(train_data)

print('DATA LOADED \n')


# tiny_data = pd.DataFrame({
#     'feature1': [0, 1, 0, 1, 0],
#     'feature2': [1, 0, 1, 0, 1]
# })
# tiny_labels = [0, 1, 0, 1, 0]

# model = DecisionTree(depth_limit=2, ig_criterion='entropy')
# model.train(tiny_data, tiny_labels)  # Should complete instantly


# print(evaluate(model, tiny_data, tiny_labels))


model = DecisionTree(10, 'entropy')
print('TREE CREATED \n')

train(model, train_data, train_labels)

# # train(model, train_data, train_labels)

print('MODEL TRAINED \n')

print(evaluate(model, train_data, train_labels))
print(' ')
print(evaluate(model, test_data, test_labels))
print(' ')
print(evaluate(model, eval_anon_data, eval_anon_labels))


train_data_prediction = model.predict(train_data)
test_data_prediction = model.predict(test_data)
eval_anon_prediction = model.predict(eval_anon_data)


print('DONE')

# train_prediction = model.predict(train_data)
# test_prediction = model.predict(test_data)
# eval_prediction = model.predict(eval_anon_data)



eval_ids = pd.read_csv('eval.id', header=None, names = ['example_id'])

train_data_prediction = pd.Series(train_data_prediction, name = "label")
test_data_prediction = pd.Series(test_data_prediction, name = "label")
eval_anon_prediction = pd.Series(eval_anon_prediction, name = "label")

train_prediction = pd.concat([eval_ids, train_data_prediction], axis=1)
train_prediction = train_prediction.fillna(0).astype(int)
train_prediction['example_id'] = train_prediction['example_id'].astype(int)
train_prediction['label'] = train_prediction['label'].astype(int)

test_prediction = pd.concat([eval_ids, test_data_prediction], axis=1)
test_prediction = test_prediction.fillna(0).astype(int)
test_prediction['example_id'] = test_prediction['example_id'].astype(int)
test_prediction['label'] = test_prediction['label'].astype(int)

eval_prediction = pd.concat([eval_ids, eval_anon_prediction], axis=1)
eval_prediction = eval_prediction.fillna(0).astype(int)
eval_prediction['example_id'] = eval_prediction['example_id'].astype(int)
eval_prediction['label'] = eval_prediction['label'].astype(int)


train_prediction.to_csv("train_prediction.csv", index = False)
test_prediction.to_csv("test_prediction.csv", index = False)
eval_prediction.to_csv("eval_prediction.csv", index = False)