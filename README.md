# Data Science Challenge

## Problem statement
Given a training dataset and a test dataset, predict the target variable for the test dataset. The constraints of the problem are as follows:
- The training dataset only contains 800 samples, so we have to be careful not to overfit the model.
- The target variable is the column named 'target'
- The target variable has a noisy behavior, which implies that it is not a direct function of the features to predict it.
- The features are the columns named 'feature_{n}' where n is an integer between 0 and 19.
- We don't have context of what the features represent, so are required to justify the use and discard of features.

## Exploratory Data Analysis

### Data description
Afer performing a simple analysis of the dataset by performing the describe method, we notice that there is no null values in the dataset, so we dont have to perform any imputation.

The standard deviation is very high for most of the features, which implies that the features are not normalized. Also the minimum and maximum values have a big distance, which also implies that the features are not normalized. The features that have a relative small standard deviation are feature_11, feature_14, feature_18, and the target variable.

We notice that the distance between the minimum and percentile 25% for all the variables is very high, which implies that we have some outliers in every feature.
Also this behavior is present in the percentile 75% and the maximum value, so we have to deal with does outliers when modeling. 

### Feature analysis

We plotted all the features in a single boxplot to understand its distribution and to spot the behavior of the outliers.

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/boxplots/boxplot_features.png "Boxplot of features 0 to 19.")

With the boxplot above we can see that the features are very disperse, so it is probably that we have to deal with some preprocessing to normalize the features. Also, as we don't have a huge ammount of data, treating with outliers could be a bad idea, so it is probably that we should use a model that is robust to outliers and another one that is not in order to compare the results and check if outliers really matters.

### Multicollinearity
When calculating the correlation matrix of the features we notice that there are no features with a high correlation between each other, so we can conclude that there is no multicollinearity in the dataset.

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/features_correlation_matrix.png "Correlation matrix of features.")

### Target analysis

We plotted the target variable using a simple boxplot and a histogram to understand its distribution and to spot any outliers.

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/boxplots/boxplot_target.png "Boxplot of target variable.")

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/histograms/histogram_target.png "Histogram of target variable.")

With the boxplot and histogram above we can see that the target variable is nearly normally distributed, and we can confirm it by calculating the skewness and kurtosis (-0.03976 and -0.5443 respectively, which are close to 0). Also, if we analize the median and mode, we can see that they are very close to each other, which also confirms that the target variable is nearly normally distributed, so we can assume that the target variable is normally distributed.

### Basic statistics
When calculating skewness and kurtosis we notice that both values are very close to 0, which confirms that the target variable is nearly normally distributed and we can proceed assuming that it follows a normal distribution.

### Feature-target correlation analysis
#### Pearson correlation
When calculating the correlation of the features with the target variable we notice that feature_2 has a high correlation, while feature_13, feature_9 and feature_11 have a moderate correlation, and the rest of the features have a correlation close to 0. This is important because we can use this information to select the most important features for our model, and discard the rest. 

#### Spearman correlation
When calculating the Spearman correlation of the features with the target variable we notice the same behavior as the Pearson correlation, where the features feature_2, feature_13, feature_9 and feature_11 seems to be the most relevant features due to their high correlation with the target variable.

### Feature selection conclusion

Based on the correlation analysis, we propose two ways of dealing with the features:
1. Use all the features, but use a model that is robust to outliers and multicollinearity.
2. Use only the most relevant features, and use a model that is not robust to outliers and multicollinearity.

## Preprocessing
We decided to apply a StandardScaler function to the features to normalize them and deal with the different ranges of each variable. The decision of using StandardScaler was based on the fact that it is a robust scaler and it is not affected by outliers, as well as it would help to implement a linear model like linear regression. The result of applying the scaler is 

We didn't apply any transformation to the target variable because it is already nearly normally distributed.

## Model Training
### ElasticNet
We decided to use ElasticNet because it would help us to understand and to interpret what features are important to predict the target variable and what variables should be discarded. To do this, we implemented an ElasticNetCV model to perform a cross-validation and find the best parameters for the model, including L1 regularization parameter. 

The hyperparameters found by the model were:

| Hyperparameter |   Value  |
|:---------------|---------:|
| alpha          | 0.086975 |
| l1_ratio       | 1.000000 |

The features selected by the model were:

| Feature    |   Coefficient |
|:-----------|--------------:|
| feature_2  |     2.67918   |
| feature_13 |     1.88828   |
| feature_9  |     1.85305   |
| feature_11 |     1.54349   |
| feature_18 |     0.106193  |
| feature_12 |     0.0953418 |
| feature_14 |     0.0554773 |
| feature_3  |     0.0442084 |
| feature_19 |    -0.0103863 |
| feature_17 |    -0.101325  |

obtaining the following metrics on the training dataset:

|       R2 |     MSE |     MAE |
|---------:|--------:|--------:|
| 0.701501 | 7.72236 | 2.15199 |

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/feature_importances/feature_importances_elastic.png "Feature importances of ElasticNet model.")

With this results in mind, we can conclude that we are cathching the linear relationship between the features and the target variable, but we are not cathching most of the information despite that the model is selecting the most relevant features, so we need to implement a model that can capture the non-linear relationships between the features and the target variable.

### Xgboost
We decided to implement a second solution using an XGBoost Regressor. The XGBoost algorithm is well known for being a robust option when working with noisy data and handling outliers. Also, XGBoost can help us to discover interactions between features, which is not possible to discover by the linear model itself. For this implementation we used the GridSearchCV function to perform a 7-fold cross-validation with 150 combinations in order to find the best parameters for the model using the R2 score as the evaluation metric.

The first trained model with this implementation was using all the features included in the dataset, and it obtained the following metrics for the training dataset:

|       R2 |     MSE |     MAE |
|---------:|--------:|--------:|
| 0.929033 | 1.835972 | 1.059891 |

However, if we analyze the cross-validation results, we obtained a R2 score of 0.832491. We consider this result as an excelent one because we are modeling the data very well and we are close to the 0.92 R2 score that was expected from the training data, preventing from catching huge ammount of noise. Despite the fact that there is a gap between the training and validation R2 score, is less than 0.1, so we consider that this result is good enough to proceed to the next step.

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/feature_importances/feature_importances_xgboost.png "Feature importances of XGBoost model.")

Analyzing the feature importances, we can see that feature_2, feature_13, feature_9 and feature_11 are the most important features, which is consistent with the correlation analysis. However, we can also see that the other features that we thought were irrelevant are actually lightly important to predict the target variable and contribute to modeling the target function, even if they have a low correlation with it. 

Getting an R2 score of 0.832491 in the cross-validation and 0.929033 in the training dataset means that out model is is able to generalize well to unseen data, so it is expected that the R2 score in the blind test dataset is close to the cross-validation R2 score.

Before creating the final model, we decided to implement a model using only the most important features to see if we could improve the R2 score. The features selected were feature_2, feature_13, feature_9 and feature_11. Pitifully, the R2 score presented an agressive decrease, obtaining quite the same performance as the linear model. With this result in mind, we can reinforce the idea that all the features that seems to be irrelevant are actually important to predict the target variable and contribute to modeling the target function due to the non-linear relationships between the features and the interactions between them.

## Predictions on Train Dataset

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/predictions/elasticnet.png "Predictions on train dataset using elasticnet.")

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/predictions/xgboost.png "Predictions on train dataset using xgboost.")

One of the most important things to visually notice is the difference between the predictions of the linear model and the XGBoost model. The scatter of the predictions of the linear model is much higher than the scatter of the predictions of the XGBoost model. This is expected because the linear model is not able to capture the non-linear relationships between the features and the target variable, while the XGBoost model is able.

## 7. Final Model Selection
Taking into account the results obtained from the past sections, we decided to use the XGBoost model as the final model because it obtained the best R2 score in the cross-validation and outperformed the linear model in all the metrics.

If we are asked for one argument to justify this decision, it would be the residuals obtained for the train dataset. 

![alt text](https://github.com/BrandonHT/Wizeline-DS-Challenge/blob/main/plots/residuals/residuals_xgb.png "Residuals on train dataset using xgboost.")

The residuals obtained from the predictions on the training dataset are more evenly distributed around 0 than the residuals obtained from the predictions on the training dataset using elasticnet. What it means is that the XGBoost is not missing too much when predicting, so the generalization of the data is really good despite we don't have a lot of data and is noisy.

## 8. Predictions on Blind Test
The predictions on the blind test dataset were generated using the XGBoost model with the best parameters found using GridSearchCV, and were saved under the folder [predictions](./predictions/).