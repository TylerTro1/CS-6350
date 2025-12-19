# Assignment 1

Functions to implement:

```
model.py
    MajorityBaseline.train()
    MajorityBaseline.predict()
    DecisionTree.train()
    DecisionTree.predict()

train.py
    train()
    evaluate()
    calculate_accuracy()

cross_validation.py
    cross_validation()
```

Once you've implemented `MajorityBaseline` and the functions in `train.py`, you can train and evaluate your model with:
```sh
python train.py -m majority_baseline
```

Make sure your code works for `MajorityBaseline` before moving on to `DecisionTree`. 

Next, once you've completed `DecisionTree`, you can train and evaluate your model with:
```sh
python train.py -m decision_tree                   # runs with no depth limiting
python train.py -m decision_tree -d 2              # runs with depth_limit set to 2
python train.py -m decision_tree -i collision      # runs with "collision entropy" as the ig_criterion instead of "entropy"
python train.py -m decision_tree -d 2 -i collision # runs with depth_limit=2 and ig_criterion="collision"
```

Once you've implemented the necessary code in `cross_validation.py`, you can run cross validation with:
```sh
python cross_validation.py -d 1 2 3 4              # runs CV with the depth_limit_values=[1, 2, 3, 4]
python cross_validation.py -d 1 2 3 4 -i collision # runs CV with the depth_limit_values=[1, 2, 3, 4] and ig_criterion="collision"
```
