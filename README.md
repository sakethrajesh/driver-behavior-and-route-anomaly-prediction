
## Classification

### multinomial_logistic_regression()
```python
Best Parameters: <bound method BaseEstimator.get_params of GridSearchCV(cv=3, estimator=LogisticRegression(), n_jobs=-1,
             param_grid={'C': array([1.00000000e-04, 2.63665090e-04, 6.95192796e-04, 1.83298071e-03,
       4.83293024e-03, 1.27427499e-02, 3.35981829e-02, 8.85866790e-02,
       2.33572147e-01, 6.15848211e-01, 1.62377674e+00, 4.28133240e+00,
       1.12883789e+01, 2.97635144e+01, 7.84759970e+01, 2.06913808e+02,
       5.45559478e+02, 1.43844989e+03, 3.79269019e+03, 1.00000000e+04]),
                         'max_iter': [5000, 10000], 'penalty': ['l2'],
                         'solver': ['newton-cg', 'lbfgs', 'sag']},
             verbose=2)>
Accuracy on test data: 0.717
```


### svn_classification()
```python
Best Parameters: {'C': 1, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy: 0.66

Classification Report:
              precision    recall  f1-score   support

           0       0.77      0.24      0.36       825
           1       0.64      0.95      0.76      1175

    accuracy                           0.66      2000
   macro avg       0.71      0.59      0.56      2000
weighted avg       0.69      0.66      0.60      2000


Confusion Matrix:
[[ 196  629]
 [  58 1117]]
 ```

 ### decision_tree
 #### post
 ```python
Best ccp_alpha: 0.004583333333333333
Train Accuracy of Best Tree: 0.9025
Test Accuracy of Best Tree: 0.6300
```

#### pre
```python
Best Parameters: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 5, 'max_features': 's
qrt', 'min_samples_split': 10, 'splitter': 'best'}
Train Accuracy - : 0.733
Test Accuracy - : 0.580

Classification Report:
              precision    recall  f1-score   support

           0       0.47      0.56      0.51        39
           1       0.68      0.59      0.63        61

    accuracy                           0.58       100
   macro avg       0.57      0.58      0.57       100
weighted avg       0.60      0.58      0.58       100


Confusion Matrix:
[[22 17]
 [25 36]]

 50000
 Best Parameters: {'ccp_alpha': 0.0, 'criterion': 'entropy', 'max_depth': 10, 'max_features'
: 'log2', 'min_samples_split': 5, 'splitter': 'random'}
Train Accuracy - : 0.716
Test Accuracy - : 0.710

Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.23      0.36      3523
           1       0.70      0.97      0.81      6477

    accuracy                           0.71     10000
   macro avg       0.75      0.60      0.59     10000
weighted avg       0.74      0.71      0.65     10000


Confusion Matrix:
[[ 822 2701]
 [ 200 6277]]

 100000
 Best Parameters: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'l
og2', 'min_samples_split': 2, 'splitter': 'best'}
Train Accuracy - : 0.743
Test Accuracy - : 0.739

Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.11      0.20      5733
           1       0.74      0.99      0.84     14267

    accuracy                           0.74     20000
   macro avg       0.79      0.55      0.52     20000
weighted avg       0.76      0.74      0.66     20000


Confusion Matrix:
[[  633  5100]
 [  121 14146]]
```