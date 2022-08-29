#!/usr/bin/env python3

import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def boston_house_price():
    """
    波士顿房价预测
    :return:
    """
    boston = load_boston()
    data = boston.data
    target = boston.target
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    gbm = lgb.LGBMRegressor(objective="regression", num_leaves=31, learning_rate=0.01, n_estimators=100)
    gbm.fit(x_train, y_train, eval_set=(x_test, y_test), eval_metric="l1", early_stopping_rounds=5)

    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)
    for val_p, val_t in zip(y_pred, y_test):
        print("%.2f, %.2f" % (val_p, val_t))

    lgb.plot_importance(gbm, max_num_features=30)
    plt.show()


if __name__ == "__main__":
    boston_house_price()
