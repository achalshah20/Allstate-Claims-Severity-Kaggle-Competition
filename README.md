# Allstate-Claims-Severity-Kaggle-Competition

**Task:** Accurately predicts claims severity.

**Approach:**

1. Exploratory data analysis to find the correlation between continuous and categorical features. In order to analyse, heatmaps, boxplots, distribution plots were generated.
2. All the features are already normalized but target variable is highly skewed so log transformation was applied.
3. Feature engineering was performed in different ways and 3 different types of feature sets were generated. i.e one hot encoding
4. Generalized linear models like ridge,lasso and elasticnet regression were performed but it did not perform very well.
5. XGboost with proper tuning performed very good on the public leaderboard
6. Kera's NN did not perform well but it performed very well with proper number of bags(5). i.e bagging was required because NN was giving results with high variance.

**This is an ongoing project so following methods are yet to be applied:**

1. Gradient boosting machine
2. Random forest regressor, Extra trees regressor
3. Support vector regressor (I think, it will take a lot of time to run because of number of records)
4. Deep neural network
5. Ensemble stacking with best models
