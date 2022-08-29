#!/usr/bin/env python3

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import lightgbm

if __name__ == "__main__":
    X, y = make_regression(n_samples=10000, n_features=10, n_informative=3, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # 建立booster
    lgb_model = lightgbm.LGBMRegressor(max_depth=6, num_leaves=50, learning_rate=0.05, random_state=42)

    # 训练模型
    lgb_model.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric="rmse")

    # 获取参数
    params = lgb_model.get_params()
    print(params)

    # 获取特征数目
    n_features = lgb_model.n_features_
    print(n_features)

    # 获取特征数目
    n_features_in = lgb_model.n_features_in_
    print(n_features_in)

    # best score
    best_score = lgb_model.best_score_
    print(best_score)

    # best iteration
    best_iteration = lgb_model.best_iteration_
    print(best_iteration)

    # 获取objective
    objective_str = lgb_model.objective_
    print(objective_str)

    # 获取booster对象
    booster_model = lgb_model.booster_
    print(booster_model)

    # 获取评价结果
    eval_results = lgb_model.evals_result_
    print(eval_results)

    # 特征重要性
    feature_importances = lgb_model.feature_importances_
    print(feature_importances)

    # 特征名字
    feature_names = lgb_model.feature_name_
    print(feature_names)

    lightgbm.plot_importance(lgb_model.booster_)
    plt.show()