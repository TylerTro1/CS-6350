from model import Perceptron, PerceptronEnsemble
from evaluate import accuracy
from cross_validation import cross_validation
from data import load_data

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.pipeline import Pipeline


import pandas as pd
import numpy as np




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

loaded_data = load_data()
cv_folds = loaded_data['cv_folds']


### REMOVE COLUMNS OF ZEROES ###
cols_to_drop = train_data.columns[(train_data == 0).all()]
train_data = train_data.drop(cols_to_drop, axis = 1)

# cols_to_drop = test_data.columns[(test_data == 0).all()]
test_data = test_data.drop(cols_to_drop, axis = 1)

# cols_to_drop = eval_anon_data.columns[(eval_anon_data == 0).all()]
eval_anon_data = eval_anon_data.drop((cols_to_drop), axis = 1)




# best_params, _ = cross_validation(cv_folds, 'decay', [10, 1, 0.1, 0.001, 0.0001, 0.00001], [10, 0.1, 0.01, 0.001, 0.0001, 0.00001], 20)
# print(best_params)

best_params = {'lr': 0.1, 'mu': 0.1}
#lr = 0.1, mu = 0.1


### VARIANCE THRESHOLD APPROACH ###
selector = VarianceThreshold(threshold = 0.01)

train_data_reduced = selector.fit_transform(train_data)
train_data_variance_threshold = pd.DataFrame(train_data_reduced, columns = train_data.columns[selector.get_support()])

# test_data_reduced = selector.fit_transform(test_data)
test_data_variance_threshold = pd.DataFrame(test_data, columns = test_data.columns[selector.get_support()])

# eval_anon_data_reduced = selector.fit_transform(eval_anon_data)
eval_anon_variance_threshold = pd.DataFrame(eval_anon_data, columns = eval_anon_data.columns[selector.get_support()])

# print(train_data)




### THRESHOLD APPROACH ###
threshold = 0.95
train_ratio = (train_data != 0).sum() / len(train_data)
test_ratio = (test_data != 0).sum() / len(test_data)
eval_anon_ratio = (eval_anon_data != 0).sum() / len(eval_anon_data)

train_data_threshold = train_data.loc[:, train_ratio > (1 - threshold)]

test_data_threshold = test_data.loc[:, test_ratio > (1 - threshold)]

eval_data_threshold = eval_anon_data.loc[:, eval_anon_ratio > (1 - threshold)]

# print(train_data)





### STANDARD SCALER APPROACH ###
standard_scaler = StandardScaler()

standard_scaler.fit(train_data)

train_data_scaled = standard_scaler.fit_transform(train_data)
train_data_standard_scaler = pd.DataFrame(train_data_scaled, columns = train_data.columns)

test_data_scaled = standard_scaler.transform(test_data)
test_data_standard_scaler = pd.DataFrame(test_data_scaled, columns = test_data.columns)

eval_data_scaled = standard_scaler.transform(eval_anon_data)
eval_data_standard_scaler = pd.DataFrame(eval_data_scaled, columns = eval_anon_data.columns)

# print(train_data)
# print(train_data.shape)





### MIN-MAX SCALER APPROACH ###
minmax_scaler = MinMaxScaler()

train_data_minmax = pd.DataFrame(minmax_scaler.fit_transform(train_data), columns=train_data.columns)
test_data_minmax = pd.DataFrame(minmax_scaler.transform(test_data), columns=test_data.columns)
eval_data_minmax = pd.DataFrame(minmax_scaler.transform(eval_anon_data), columns=eval_anon_data.columns)





### PCA APPROACH ###
pca = PCA(n_components=0.95)  # keep 95%
train_data_pca = pd.DataFrame(pca.fit_transform(train_data))
test_data_pca = pd.DataFrame(pca.transform(test_data))
eval_data_pca = pd.DataFrame(pca.transform(eval_anon_data))

# print(train_data.shape)




### POLYNOMIAL FEATURES ###
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

train_data_poly = pd.DataFrame(poly.fit_transform(train_data))
test_data_poly = pd.DataFrame(poly.transform(test_data))
eval_data_poly = pd.DataFrame(poly.transform(eval_anon_data))


print(train_data.shape)




#SelectKBest
k_best_selector = SelectKBest(score_func = f_classif, k=50)
train_selected = selector.fit_transform(train_data_standard_scaler, train_labels)
test_selected = selector.transform(test_data_standard_scaler)
eval_selected = selector.transform(eval_data_standard_scaler)



### COMBINE PRE-PROCESSING ###
vt = VarianceThreshold(threshold=0.05)
pca = PCA(n_components=50)

# Apply transformations
X_scaled = standard_scaler.fit_transform(train_data)
X_vt = vt.fit_transform(X_scaled)
X_pca = pca.fit_transform(X_vt)

# Train Perceptron
model = PerceptronEnsemble(num_features=X_pca.shape[1], estimator_count= 4, lr_values = [1, 0.1, 0.0001, 0.001], decay_bools = [True, False, True, False], mu_values = [1, 1, 1, 1])
model.train(X_pca, train_labels.to_numpy(), 20)

# Predict on eval set
X_eval_scaled = pca.transform(vt.transform(standard_scaler.transform(eval_anon_data)))
X_test_scaled = pca.transform(vt.transform(standard_scaler.transform(test_data)))
y_test_pred = model.predict(X_test_scaled)
y_pred = model.predict(X_eval_scaled)

print("test_labels:", test_labels.shape)
print("y_test_pred:", y_test_pred.shape)


print(f'F1-Score Test: {f1_score(test_labels, y_test_pred)}')
print(f'F1-Score Eval: {f1_score(eval_anon_labels, y_pred)}')


# model = Perceptron(train_data.shape[1], best_params['lr'], True, best_params['mu'])
# model.train(train_data.to_numpy(), train_labels.to_numpy(), 20)

# test_preds = model.predict(test_data.to_numpy())
# eval_preds = model.predict(eval_anon_data.to_numpy())

# print(f'F1-Score: {f1_score(test_labels, test_preds)}')
# print(f'F1-Score Eval: {f1_score(eval_anon_labels, eval_preds)}')










eval_anon_prediction = pd.Series(y_pred, name = "label")

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