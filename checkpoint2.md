# US Faculty Employment Potential
### Quantifying hiring bias for tenure-track faculty employments
### Project Checkpoint 2
##### 25 November 2022
##### Machine Learning Ninjas: Bhatia, Ghosh, Nwogu

---

## Outline
- Preliminary ML Pipeline
- Final ML Pipeline
- ML Pipeline - Input
- ML Modelling and Results
- Potential Issues
- Potential Improved Methods for Enhanced Performance
- Future Plan


---

## Preliminary ML Pipeline 

`$$AttritionProbability = \frac{AttritionEvents}{AttritionEvents + NonAttritionEvents}$$`



![Planned ML System Pipeline](https://mermaid.ink/img/pako:eNp1Uk1vwjAM_StVzvAHepiEVjRVYoAK2g5ND6ExJSIfKHXYEOK_L02h6mjpyX32e7GffSWl4UBispfmpzwwi1G0yKimOOcVLESN0XT6FqX65JDqyH-prlGgQ2H0BhnWw7wPAziTlbECD6qR635CauUwECiG8jwxigkdUS-RQGUBeq-kScDfnbWgcZgYENYWfFxBxvTxBXVQ8gGagw3hDNE36osyhhCQHrEhFP_muWbN-3Xtky1dWlM1iVtT1g6ad5pra3ZsJ6TAS9H69fA57wz3Ko3Qlv0abdTli0k36CPlz8iSKXhhIB_HO8bWIJMh-oR2iG-jQBfjC88HF3BveFT6PsQCziCj8bGWRnf-zM9-UfX_PfQwbx935WMRd-hplSvLhWayjxdkQhRYf2Pcn_q1mYsSPIACSmIfcmaPlFB983XModlcdElitA4mxJ24v4NEsMoyReI9kzXc_gB5nS7q?type=png)

Notes:

- The initial goal was to obtain `AttritionProbability` as the output from the ML Algortihm. This variable represents what a professor's succes is defined by. A lower `AttritionProbability` is the predictor label that indicates greater/higher success while a lower value indicates otherwise.  This is still our goal.

- Our initial plan was to combine EdgeList and InstitutionStats into a new input dataframe with eight features.  Each row contained data from multiple professors and our initial plan was to use `Total` number of professors as the weight for each data point.  However, since the male and female breakdown numbers were scattered in separate fields, we could not just apply weights to each data point.  It would have really complicated our pipeline.

- So instead of that, we exploded the data points with a new categorical field `Gender`.  We also dropped several features.

---


## Final ML Pipeline


![Final Pipeline Flowchart](https://mermaid.ink/img/pako:eNqVVEtOwzAQvUrkNVygCyREK1SpQNUiWMRdTOMhtXBs5IwLVdW7YzsmtLgBkdX4ze_NL3tWGYFsxF6Uea82YKkoZguuOU1EjTPZUnF5eVUssEWw1QZtD031myOuC_9NdUuSHEmjlwTURv2NsxY1ndX95TvG2iL-7erFCF6r2lhJmyYQ7x9R9eAoOnCK5uV3JWMgKLjX-Gi3qAXa8CrmFn3aGhegX1MNGd7xi_DYNCB1FK-JfF7Pdm7NGtZSSdqtThjtF8Gxbb1N56GsqYPiEMw6quVAmFDx6RjKH1NJtRy1bSpimkf4MNo0uydQLrH-2eBk2fUhZfvagLJfhZQhj5dHOkLuofk9a4b3Ho-GQEXpDruWPZsG9er87pTZxuQt-Q6dipjhFtVAm-6N7qcx2fpNaE8HfYT5YQlXBTAsSLYxEXiwQmpQx3iqZOBYyqEj-seo_1VDRuz8JZYDB5poDc05J3eSkV2wBq0_J-H_R_uQnzPaYIOcjbwowL5yxvXB24Ejs9zpio3IOrxg7k0A4VhCbaFhoxdQLR4-AR1gypI?type=png)

Notes:

EdgeList contains aggregated data of all professors grouped by their source instutition, current institution, and specific domain or field.  Number of professors per row is indicated by the `Total` column further broken down into two `Gender` columns.  We disaggregate this data by exploding the rows.  Then we left join this dataframe with CurrentInstitutionStats and DegreeInstitutionStats to get `ResearcherData`.  `ResearcherData` is then fed through an SVD algorithm to train a model to predict `AttritionProbability`.

---

## Input

- The input for the final ML pipeline includes: 
    - Gender
    - PrestigeRankCurrent
    - PrestigeRankDegree
    - Domain
    - AttritionProbability




Notes:
The input data utilized in our final ML pipeline are
- Gender: This is a String. It represents The gender of a faculty member
- PrestigeRankCurrent: This is a Float. The SpringRank of the academic institution where the faculty is employed, scaled from 0-1. A rank of 0 indicates high prestige, a rank of 1 indicates low prestige
- PrestigeRankDegree: This is a Float. The SpringRank of the academic institution that produced the faculty, scaled from 0-1. A rank of 0 indicates high prestige, a rank of 1 indicates low prestige
- Domain: This reresents the academic domain of a faculty member
- AttritionProbability: This is a Float. It represents the probability of attrition for a given faculty member

--



![Input Data](image-20221117-232146.png)

```
    Int64Index: 680649 entries, 0 to 900819
    Data columns (total 5 columns):
    #   Column                Non-Null Count   Dtype   
    ---  ------                --------------   -----   
    0   Gender                680649 non-null  category
    1   PrestigeRankCurrent   680649 non-null  float64 
    2   PrestigeRankDegree    680649 non-null  float64 
    3   Domain                680649 non-null  category
    4   AttritionProbability  680649 non-null  float64 
    dtypes: category(2), float64(3)
    memory usage: 22.1 MB
```

## Metrics
- Accuracy
    - Measures model accuracy

- Mean Squared Error (MSE)
    - Measures the amount of error in the model 
    - Average squared difference between observed and predicted values

- R-squared
    - Goodness-of-fit metric 
    - Percentage of the variance in the dependent variable that the independent variables explain collectively

Notes:
 
--

## ML Modelling 
- Regression Analysis
    - Continuous prediction target variable
    - Nature of project 
    - Machine Learning Pipeline
- Regressor Models:
    - Decision Tree Regressor
    - Random Forest Regressor
    - Stochastic Gradient Descent Regressor
    - Catboost Regressor
    - XGBoost Regressor
    - Lasso Regressor 
    - Ridge Regressor


Notes:
The variable we are trying to predict is a continuous one, which makes regression analysis the efficient modelling algorithm to use. Also, due to the nature of the project, specifically using a set of independent variables to predict a dependent variable (AttritionProbability), the best type of analysis is regression. This is also reflected in the ML Pipeline
 
--

### Stochastic Gradient Descent Regressor
- Used with a `max_iter` value of 10000000
- Used with an `tol` value of 1e-10

```
SGD = make_pipeline(StandardScaler(),SGDRegressor(max_iter=10000000, tol=1e-10))
SGD.fit(train_X,train_Y)
pred_Y = SGD.predict(test_X)
accuracy_SGD = round(SGD.score(train_X, train_Y) * 100, 2)
accuracy_SGD, mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)
```

Notes:
 
--

### Stochastic Gradient Descent Regressor - Results
- Accuracy: 14.53

- Mean Squared Error (MSE): 0.0031783664239967953

- R-squared: 0.15377511450664805

![Stochastic Gradient Descent Regressor Results](reg_results/sgd_results.png)

Notes:
 
--

### Lasso Regressor

- Used with a `alpha` value of 0.0001

```
lasso = linear_model.Lasso(alpha=0.0001)
lasso.fit(train_X,train_Y)
pred_Y = lasso.predict(test_X)
accuracy_lasso = round(lasso.score(train_X, train_Y) * 100, 2)
accuracy_lasso, mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)
```

Notes:
 
--

### Lasso Regressor - Results
- Accuracy: 14.65

- Mean Squared Error (MSE): 0.0031764477499716427

- R-squared: 0.15428595230526543

![Lasso Regressor Results](reg_results/lasso_results.png)

Notes:
 
--

### Ridge Regressor

- Used with a `alpha` value of 0.01

```
ridge = Ridge(alpha=.01)
ridge.fit(train_X,train_Y)
pred_Y = ridge.predict(test_X)
accuracy_ridge = round(ridge.score(train_X, train_Y) * 100, 2)
accuracy_ridge, mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)
```

Notes:
 
--

### Ridge Regressor - Results
- Accuracy: 14.69

- Mean Squared Error (MSE): 0.003174582604292563

- R-squared: 0.15478253843730827

![Ridge Regressor Results](reg_results/ridge_results.png)

Notes:
 
---

### Catboost Regressor
- Used with a `verbose` value of 0

```
catboost = CatBoostRegressor(verbose=0)
catboost.fit(train_X,train_Y)
pred_Y = catboost.predict(test_X)
accuracy_catboost = round(catboost.score(train_X, train_Y) * 100, 2)
accuracy_catboost, mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)
```

Notes:
 
--

### Catboost Regressor Results
- Accuracy: 38.5

- Mean Squared Error (MSE): 0.00235755390490074

- R-squared: 0.3723125288020457

![Catboost Regressor Results](reg_results/catboost_results.png)

Notes:
 
--

### XGBoost Regressor

```
xgboost = XGBRegressor()
xgboost.fit(train_X,train_Y)
pred_Y = xgboost.predict(test_X)
accuracy_xgboost = round(xgboost.score(train_X, train_Y) * 100, 2)
accuracy_xgboost, mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)
```

Notes:
 
--

### XGBoost Regressor Results
- Accuracy: 56.0

- Mean Squared Error (MSE): 0.0017033830717451874

- R-squared: 0.5464823898352575

![XGBoost Regressor Results](reg_results/xgboost_results.png)

Notes:
 
--

### Decision Tree Regressor
- Used with a `random_state` value of 42

```
DT = DecisionTreeRegressor(random_state=42)
DT.fit(train_X,train_Y)
pred_Y = DT.predict(test_X)
accuracy_DT = round(DT.score(train_X, train_Y) * 100, 2)
accuracy_DT, mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)
```


Notes:
 
--

### Decision Tree Regressor - Results

- Accuracy: 99.93

- Mean Squared Error (MSE): 7.477459511247769e-05

- R-squared: 0.9800916210575562

![Decision Tree Regressor Results](reg_results/dt_results.png)

Notes:
 
--

### Random Forest Regressor
- Used with a `random_state` value of 42
- Used with an `n_jobs` value of -1

```
RF = RandomForestRegressor(random_state=42, n_jobs=-1)
RF.fit(train_X,train_Y)
pred_Y = RF.predict(test_X)
accuracy_RF = round(RF.score(train_X, train_Y) * 100, 2)
accuracy_RF, mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)
```

Notes:
 
--

### Random Forest Regressor - Results

- Accuracy: 99.84

- Mean Squared Error (MSE): 6.074051620977429e-05

- R-squared: 0.9838281275606393

![Random Forest Regressor Results](reg_results/rf_results.png)

Notes:
 
--

## Modelling Results Summary

- Decision Tree and Random Forest regressors had highest accuracy values
- Lasso, Ridge, and Stochastic Gradient Descent regrssors had lowest accuracy values
- Catboost and XGBoost had average accuracy values

Notes:
 
---

## Model Improvement Method

### OneHot Encoding - `Gender` category
- Some models required OneHot encoding
- Others (e.g. CatBoost) handle it internally
 
### Weighted Data Points
-  `Count` as the weight for each data point

Notes:
 
---

## Issues

- CatBoost and XGBoost work well with unbalanced data
- Some models prefer categorical data and handle One Hot Encoding internally

Notes:


---

## Future Improvements

- No access to original data
- Not enough information to derive appropriate correlations - e.g. publications, citations

- Code cleanup / optimization, e.g.
```py
EdgeList['Gender'] = EdgeList.apply(lambda x: int(x['GenderUnknown'])*['U']+int(x['Men'])*['M']+int(x['Women'])*['F'], axis=1)
```

- Compute PrestigeRank: Calculate (PrestigeRank, AttritionEvents, NonAttritionEvents) from other data points within the same field/domain weighted by AttritionEvents + NonAttritionEvents.  If none of that is present, consider calculating from remaining data points.

Notes:


---

 ## Lessons Learned



Notes:Exploding across Gender will make a lot of identical rows
Some of these rows will be identical across train_df and test_df if we split it beforehand.
So, we split it into train_df and test_df first after all the feature engineering.

And then we explode the aggregated data to make sure that there are no data point repeats between the training dataset and testing dataset


Catboost and xgboost work much better for unbalanced data. Also, catboost does not work well for one-hot encoded data, however in this case, Gender information is important, so we have to one-hot encode it, to make interpretations.

Regarding regression models, Random Forest works the best. After 10 fold cross validation, we get an accuracy 98.01% without any overfitting issues.

For regression models, residual plots make sense, because visualizing the residuals help us see the how well our model performed.




<a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=0fa8eae6-cbdb-42f3-8c8f-0479a8e45918' target="_blank">
<img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
