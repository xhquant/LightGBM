#!/usr/bin/env python3

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def iris():
    iris_data = load_iris()
    data = iris_data.data
    target = iris_data.target

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    print("训练集数据: ", len(x_train))
    print("测试集数据: ", len(x_test))

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    params = {'task': 'train',
              'boosting_type': 'gbdt',  # 设置提升类型
              'objective': 'regression',  # 目标函数
              'metric': {'l2', 'auc'},  # 评估函数
              'num_leaves': 31,  # 叶子节点数
              'learning_rate': 0.05,  # 学习速率
              'feature_fraction': 0.9,  # 建树的特征选择比例
              'bagging_fraction': 0.8,  # 建树的样本采样比例
              'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
              'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
              }
    gbm = lgb.train(params=params, train_set=lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

    for val_p, val_t in zip(y_pred, y_test):
        print(val_p, val_t)


if __name__ == "__main__":
    iris()
