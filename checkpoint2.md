# US Faculty Employment Potential
### Quantifying hiring bias for tenure-track faculty employments
### Project Checkpoint 2
##### 25 November 2022
##### Machine Learning Ninjas: Bhatia, Ghosh, Nwogu

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/intro.ogg"></audio>

---

## Outline
- Preliminary ML Pipeline
- Final ML Pipeline
- ML Pipeline - Input
- ML Modelling and Results
- Models Summary and Issues Encountered
- Potential Improved Methods for Enhanced Performance
- Future Plan

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/outline.m4a"></audio>

Note: This presentation is not a comprehensive product and only covers our key findings.  We have discussed everything listed here in greater detail in our final report.

---

## Preliminary ML Pipeline 

`$$AttritionProbability = \frac{AttritionEvents}{AttritionEvents + NonAttritionEvents}$$`



![Planned ML System Pipeline](https://mermaid.ink/img/pako:eNp1Uk1vwjAM_StVzvAHepiEVjRVYoAK2g5ND6ExJSIfKHXYEOK_L02h6mjpyX32e7GffSWl4UBispfmpzwwi1G0yKimOOcVLESN0XT6FqX65JDqyH-prlGgQ2H0BhnWw7wPAziTlbECD6qR635CauUwECiG8jwxigkdUS-RQGUBeq-kScDfnbWgcZgYENYWfFxBxvTxBXVQ8gGagw3hDNE36osyhhCQHrEhFP_muWbN-3Xtky1dWlM1iVtT1g6ad5pra3ZsJ6TAS9H69fA57wz3Ko3Qlv0abdTli0k36CPlz8iSKXhhIB_HO8bWIJMh-oR2iG-jQBfjC88HF3BveFT6PsQCziCj8bGWRnf-zM9-UfX_PfQwbx935WMRd-hplSvLhWayjxdkQhRYf2Pcn_q1mYsSPIACSmIfcmaPlFB983XModlcdElitA4mxJ24v4NEsMoyReI9kzXc_gB5nS7q?type=png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/pre-pipeline.m4a"></audio>

Notes:

- The initial goal was to obtain `AttritionProbability` as the output from the ML Algortihm. This variable represents what a professor's succes is defined by. A lower `AttritionProbability` is the predictor label that indicates greater/higher success while a lower value indicates otherwise.  This is still our goal.

- Our initial plan was to combine EdgeList and InstitutionStats into a new input dataframe with eight features.  Each row contained data from multiple professors and our initial plan was to use `Total` number of professors as the weight for each data point.  However, since the male and female breakdown numbers were scattered in separate fields, we could not just apply weights to each data point.  It would have really complicated our pipeline.

- So instead of that, we exploded the data points with a new categorical field `Gender`.  We also dropped several features.

---


## Final ML Pipeline


![Final Pipeline Flowchart](https://mermaid.ink/img/pako:eNqVVEtOwzAQvUrkNVygCyREK1SpQNUiWMRdTOMhtXBs5IwLVdW7YzsmtLgBkdX4ze_NL3tWGYFsxF6Uea82YKkoZguuOU1EjTPZUnF5eVUssEWw1QZtD031myOuC_9NdUuSHEmjlwTURv2NsxY1ndX95TvG2iL-7erFCF6r2lhJmyYQ7x9R9eAoOnCK5uV3JWMgKLjX-Gi3qAXa8CrmFn3aGhegX1MNGd7xi_DYNCB1FK-JfF7Pdm7NGtZSSdqtThjtF8Gxbb1N56GsqYPiEMw6quVAmFDx6RjKH1NJtRy1bSpimkf4MNo0uydQLrH-2eBk2fUhZfvagLJfhZQhj5dHOkLuofk9a4b3Ho-GQEXpDruWPZsG9er87pTZxuQt-Q6dipjhFtVAm-6N7qcx2fpNaE8HfYT5YQlXBTAsSLYxEXiwQmpQx3iqZOBYyqEj-seo_1VDRuz8JZYDB5poDc05J3eSkV2wBq0_J-H_R_uQnzPaYIOcjbwowL5yxvXB24Ejs9zpio3IOrxg7k0A4VhCbaFhoxdQLR4-AR1gypI?type=png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/fin-pipeline.m4a"></audio>

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

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/input.m4a"></audio>

Notes:

The input data utilized in our final ML pipeline are
- Gender: This is a String. It represents The gender of a faculty member
- PrestigeRankCurrent: Current university and department. This is a Float. The SpringRank of the academic institution where the faculty is employed, scaled from 0-1. A rank of 0 indicates high prestige, a rank of 1 indicates low prestige. 
- PrestigeRankDegree: University and department that granted Ph.D. This is a Float. The SpringRank of the academic institution that produced the faculty, scaled from 0-1. A rank of 0 indicates high prestige, a rank of 1 indicates low prestige
- Domain: This reresents the academic domain of a faculty member
- AttritionProbability: This is a Float. It represents the probability of attrition for a given faculty member

--



![Picture title](image-20221117-232146.png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/input-2.m4a"></audio>

---

## Metrics
- Accuracy
    - Measures model accuracy

- Mean Squared Error (MSE)
    - Measures the amount of error in the model 
    - Average squared difference between observed and predicted values

- R-squared
    - Goodness-of-fit metric 
    - Percentage of the variance in the dependent variable that the independent variables explain collectively

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/metrics.m4a"></audio>

Notes:
 
---

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

- Residual Plots used to visualize model performances

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/ml-modelling.m4a"></audio>

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

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/sgd-1.m4a"></audio>

Notes:
 
--

### Stochastic Gradient Descent Regressor - Results
- Accuracy: 14.53

- Mean Squared Error (MSE): 0.0031783664239967953

- R-squared: 0.15377511450664805

![Stochastic Gradient Descent Regressor Results](reg_results/sgd_results.png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/sgd-2.m4a"></audio>

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

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/lasso-1.m4a"></audio>

Notes:
 
--

### Lasso Regressor - Results
- Accuracy: 14.65

- Mean Squared Error (MSE): 0.0031764477499716427

- R-squared: 0.15428595230526543

![Lasso Regressor Results](reg_results/lasso_results.png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/lasso-2.m4a"></audio>

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

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/ridge-1.m4a"></audio>

Notes:
 
--

### Ridge Regressor - Results
- Accuracy: 14.69

- Mean Squared Error (MSE): 0.003174582604292563

- R-squared: 0.15478253843730827

![Ridge Regressor Results](reg_results/ridge_results.png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/ridge-2.m4a"></audio>

Notes:
 
--

### Catboost Regressor
- Used with a `verbose` value of 0

```
catboost = CatBoostRegressor(verbose=0)
catboost.fit(train_X,train_Y)
pred_Y = catboost.predict(test_X)
accuracy_catboost = round(catboost.score(train_X, train_Y) * 100, 2)
accuracy_catboost, mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)
```

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/cat-1.m4a"></audio>

Notes:
 
--

### Catboost Regressor Results
- Accuracy: 38.5

- Mean Squared Error (MSE): 0.00235755390490074

- R-squared: 0.3723125288020457

![Catboost Regressor Results](reg_results/catboost_results.png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/cat-2.m4a"></audio>

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

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/xg-1.m4a"></audio>

Notes:
 
--

### XGBoost Regressor Results
- Accuracy: 56.0

- Mean Squared Error (MSE): 0.0017033830717451874

- R-squared: 0.5464823898352575

![XGBoost Regressor Results](reg_results/xgboost_results.png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/xg-2.m4a"></audio>


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

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/dt-1.m4a"></audio>

Notes:
 
--

### Decision Tree Regressor - Results

- Accuracy: 99.93

- Mean Squared Error (MSE): 7.477459511247769e-05

- R-squared: 0.9800916210575562

![Decision Tree Regressor Results](reg_results/dt_results.png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/dt-2.m4a"></audio>

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

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/rf-1.m4a"></audio>

Notes:
 
--

### Random Forest Regressor - Results

- Accuracy: 99.84

- Mean Squared Error (MSE): 6.074051620977429e-05

- R-squared: 0.9838281275606393

![Random Forest Regressor Results](reg_results/rf_results.png)

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/rf-2.m4a"></audio>

Notes:
 
---

## Modelling Results Summary

Best Model: Random Forest

- DT and RF had highest accuracies
    - RF had lower $r^2$ error
    - Best results with 10-fold Cross Validation to avoid overfitting
- CatBoost and XGBoost had average accuracy values
    - Can be improved by tailoring input to these algorithms
    - Support GPU processing; opportunity for hyperparameter tuning
- Lasso, Ridge, and Stochastic Gradient Descent had lowest accuracy values
    - We found these algorithms to not be suitable for our use case

We have discussed each of these in much greater detail in our final report.

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/mrsummary.aac"></audio>

Notes:
Summarizing

- The best model for fitting our dataset was the Random Forest Regressor. Random Forest had the least r_square error. Also, Random Forest gives us the best results after performing 10-fold cross validation to avoid overfitting.

- Catboost and XGBoost had average accuracy values because they work much better for datasets which are unbalanced. However, for our dataset, the target prediction of Atrrition Probability does not have an uneven distribution of observations. So, the boosting models do not have a good performance on our dataset.

- Lasso, Ridge and Stochastic Gradient Descent do not perform well in our dataset because they are not suitable for our use case.
 
---

## Model Improvement Methods

### Disaggregation

- Data sourced from another study
- 885k data points combined into 405k based on common factors
    - Gender
    - Current university and department
    - University and department that granted Ph.D.
- We used Disaggregation to reproduce data in original format
    - Minimal information loss - we preserved most of the relevant data

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/mi-disagg.aac"></audio>

Notes: 

- Exploding across Gender will make a lot of identical rows

- Some of these rows will be identical across train_df and test_df if we split it beforehand.

- So, we split it into train_df and test_df first after all the feature engineering.

--

### OneHot Encoding - `Gender` category
- Some models required OneHot encoding
- Others (e.g. CatBoost) discouraged it and instead handle it internally


Also, catboost does not work well for one-hot encoded data, however in this case, Gender information is important, so we have to one-hot encode it, to make interpretations.
Subsequently, we explode the aggregated data to make sure that there are no data point repeats between the training dataset and testing dataset

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/mi-one.aac"></audio>

Notes:

- Catboost does not work well for one-hot encoded data, however in this case, Gender information is important, so we have to one-hot encode it, to make interpretations.

- And then we explode the aggregated data to make sure that there are no data point repeats between the training dataset and testing dataset.

--
 
### Weighted Data Points
- Disaggregation increased the number of data points
- Instead, we could use `Count` as the weight for each data point

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/mi-weight.aac"></audio>

Notes:

- We disaggregated the anonymized data on the basis of gender. So, we increased the number of data points. We can look at using 'Count' as the weight for each data point with respect to gender. Implementing the weights in our model can be one more approach.
---

## Future Improvements

- Raw data access
    - We had aggregated statistics
    - Disaggregated it to reproduce the data
    - Inevitable information loss

- Gather more information to derive appropriate correlations
     - Professor success more likely a function of publications, citations, h-index.
     - Connectivity make a big difference too!
     - Compare the impact of each of these factors

<audio controls data-autoplay src="https://cap5610-fall2022-ml-ninjas.github.io/checkpoint-2/audio/fut-imp.aac"></audio>

Notes:

- We did not have access to the original data but only the aggregated statistics.  We disaggregated it to reproduce the data in the original format but our underlying dataset suffered information loss in the process, which was inevitable.

- We did not have enough information to derive appropriate correlations - e.g. publications, citations, h-index.

- We can calculate (PrestigeRank, AttritionEvents, NonAttritionEvents) from other data points within the same field/domain weighted by AttritionEvents + NonAttritionEvents and compute PrestigeRange.  If none of that is present, consider calculating from remaining data points.

