#!/usr/bin/env python3

import lightgbm
import xgboost
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    X, y = make_regression(n_samples=10000, n_features=30, n_informative=5, n_targets=1, random_state=98)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    lgb_model = lightgbm.LGBMRegressor(max_depth=6, num_leaves=50, learning_rate=0.05).fit(X=x_train, y=y_train)
    lgb_model._Booster.save_model("../models/lightgbm.txt")

    xgb_model = xgboost.XGBRegressor(max_depth=6, max_leaves=50, learning_rate=0.05).fit(X=x_train, y=y_train)
    xgb_model.get_booster().dump_model("../models/xgboost.json")

    # 预测
    lgb_predict = lgb_model.predict(x_test)
    xgb_predict = xgb_model.predict(x_test)
    print(mean_squared_error(y_true=y_test, y_pred=lgb_predict))
    print(mean_squared_error(y_true=y_test, y_pred=xgb_predict))

    print(x_test[0])
    print("lightgbm predict:{0}, real:{1}".format(lgb_model.predict(x_test)[0], y_test[0]))
    print("xgboost predict:{0}, real:{1}".format(xgb_model.predict(x_test)[0], y_test[0]))
