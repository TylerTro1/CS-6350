# from model import LogisticRegression, Model, SupportVectorMachine, MODEL_OPTIONS
# from data import load_data, DATASET_OPTIONS
# from cross_validation import cross_validation
# # from evaluate import accuracy

import pandas as pd
import numpy as np
from model import DecisionTree
from cross_validation import cross_validation
from sklearn.metrics import f1_score






### LOAD DATA ###
train_data = pd.read_csv(r"D:\CS 6350\project_data\data\train.csv")
test_data = pd.read_csv(r"D:\CS 6350\project_data\data\test.csv")
eval_anon_data = pd.read_csv(r"D:\CS 6350\project_data\data\eval.anon.csv")

train_labels = train_data['label']
test_labels = test_data['label']
eval_anon_labels = eval_anon_data['label']

train_data = train_data.drop(columns = 'label')
test_data = test_data.drop(columns = 'label')
eval_anon_data = eval_anon_data.drop(columns = 'label')

eval_ids = pd.read_csv(r'D:\CS 6350\project_data\data\eval.id', header=None, names = ['example_id'])



# shuffled = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# # Split into 5 roughly equal folds
# folds = np.array_split(shuffled, 5)

# a, b = cross_validation(folds, np.arange(20), 'entropy')


# print(f'Hyperparams: {a}')

### Best Hyperparameters found a depth limit of 13



model = DecisionTree(13)

model.train(train_data, train_labels.to_list())
test_predictions = model.predict(train_data)
eval_predictions = model.predict(eval_anon_data)

print(f'F1-Score: {f1_score(eval_anon_labels, eval_predictions)}')





# END OF TESTING #
########################################################################################################################################################################################################
########################################################################################################################################################################################################
########################################################################################################################################################################################################

### Package everything into a .csv file for submission ####
    
# test_data_prediction = pd.Series(y_pred, name = "label")
eval_anon_prediction = pd.Series(eval_predictions, name = "label")

# test_prediction = pd.concat([eval_ids, test_data_prediction], axis=1)
# test_prediction = test_prediction.fillna(0).astype(int)
# test_prediction['example_id'] = test_prediction['example_id'].astype(int)
# test_prediction['label'] = test_prediction['label'].astype(int)

eval_prediction = pd.concat([eval_ids, eval_anon_prediction], axis=1)
eval_prediction = eval_prediction.fillna(1).astype(int)
eval_prediction['example_id'] = eval_prediction['example_id'].astype(int)
eval_prediction['label'] = eval_prediction['label'].astype(int)


# test_prediction.to_csv("test_prediction.csv", index = False)
eval_prediction.to_csv("eval_prediction.csv", index = False)