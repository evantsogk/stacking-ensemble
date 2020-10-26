import numpy as np
import pandas as pd
from utils import rmsle_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, \
    VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

"""This module contains code that was used during the process of building the final solution.
"""

_author_ = "Evangelos Tsogkas p3150185"


def KNN_CV(x_train, y_train, x_test):
    knn = KNeighborsRegressor(n_jobs=5)
    parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
    knn_cv = GridSearchCV(knn, parameters, scoring=make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    knn_cv.fit(x_train, y_train)
    y_pred = knn_cv.predict(x_test)

    return y_pred, knn_cv.best_params_


def DecisionTree_CV(x_train, y_train, x_test):
    dt = DecisionTreeRegressor(random_state=0)
    parameters = {'max_depth': [12, 13, 14, 15, 16, 17, 18, 19], 'min_samples_leaf': [4, 5, 6, 7, 8]}
    dt_cv = GridSearchCV(dt, parameters, make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    dt_cv.fit(x_train, y_train)
    y_pred = dt_cv.predict(x_test)

    return y_pred, dt_cv.best_params_


def RandomForest_CV(x_train, y_train, x_test):
    rf = RandomForestRegressor(n_jobs=5, random_state=0)
    parameters = {'n_estimators': [200, 300, 400], 'max_depth': [26, 27, 28, 29]}
    rf_cv = GridSearchCV(rf, parameters, make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    rf_cv.fit(x_train, y_train)
    y_pred = rf_cv.predict(x_test)

    return y_pred, rf_cv.best_params_


def ExtraTrees_CV(x_train, y_train, x_test):
    extra = ExtraTreesRegressor(n_jobs=5, random_state=0)
    parameters = {'n_estimators': [300, 400, 500], 'max_depth': [25, 26, 27, 28]}
    extra_cv = GridSearchCV(extra, parameters, make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    extra_cv.fit(x_train, y_train)
    y_pred = extra_cv.predict(x_test)

    return y_pred, extra_cv.best_params_


def AdaBoost_CV(x_train, y_train, x_test):
    ada_base = DecisionTreeRegressor(random_state=0, max_depth=14, min_samples_leaf=6)
    adaboost = AdaBoostRegressor(base_estimator=ada_base, random_state=0)

    parameters = {'n_estimators': [30, 40, 50, 60], 'learning_rate': [0.3, 0.4, 0.5, 0.6, 0.7],
                  'loss': ['linear', 'square', 'exponential']}
    adaboost_cv = GridSearchCV(adaboost, parameters, make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    adaboost_cv.fit(x_train, y_train)
    y_pred = adaboost_cv.predict(x_test)

    return y_pred, adaboost_cv.best_params_


def GradientBoosting_CV(x_train, y_train, x_test):
    gbdt = GradientBoostingRegressor(random_state=0, tol=1e-6, loss='lad')
    parameters = {'n_estimators': [80, 100, 120], 'subsample': [0.7, 0.8, 0.9], 'max_depth': [11, 12, 13]}
    gbdt_cv = GridSearchCV(gbdt, parameters, make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    gbdt_cv.fit(x_train, y_train)
    y_pred = gbdt_cv.predict(x_test)

    return y_pred, gbdt_cv.best_params_


def XGBoost_CV(x_train, y_train, x_test):
    xgb = XGBRegressor(n_jobs=5, random_state=0, objective='reg:squarederror', booster='dart')
    parameters = {'n_estimators': [40, 50, 60],
                  'subsample': [0.6, 0.7, 0.8],
                  'max_depth': [16, 17, 18],
                  'reg_lambda': [5.5, 6, 6.5]}
    xgb_cv = GridSearchCV(xgb, parameters, make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    xgb_cv.fit(x_train.values, y_train.values)
    y_pred = xgb_cv.predict(x_test.values)

    return y_pred, xgb_cv.best_params_


def LightGBM_CV(x_train, y_train, x_test):
    lgbm = LGBMRegressor(random_state=0, n_jobs=5, min_child_samples=0, boosting_type='dart')
    parameters = {'n_estimators': [150, 250, 350], 'subsample': [0.1, 0.2, 0.3], 'num_leaves': [500, 550, 600],
                  'reg_lambda': [3, 4, 5, 6]}
    lgbm_cv = GridSearchCV(lgbm, parameters, make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    lgbm_cv.fit(x_train, y_train)
    y_pred = lgbm_cv.predict(x_test)

    return y_pred, lgbm_cv.best_params_


def Voting_CV(x_train, y_train, x_test):
    extra = ExtraTreesRegressor(n_jobs=5, random_state=0, n_estimators=500, max_depth=26)
    gbdt = GradientBoostingRegressor(random_state=0, tol=1e-6, loss='lad', n_estimators=80, subsample=0.8, max_depth=13)
    xgb = XGBRegressor(n_jobs=5, random_state=0, objective='reg:squarederror', n_estimators=60, subsample=0.7,
                       max_depth=15, reg_lambda=6.5)
    xgbdart = XGBRegressor(n_jobs=5, random_state=0, objective='reg:squarederror', booster='dart', n_estimators=50,
                           subsample=0.7, max_depth=17, reg_lambda=6)
    lgbm = LGBMRegressor(random_state=0, n_jobs=5, min_child_samples=0, boosting_type='dart', n_estimators=350,
                         subsample=0.1, num_leaves=550, reg_lambda=6)

    estimators1 = [('gbdt', gbdt), ('xgb', xgb), ('xgbdart', xgbdart), ('lgbm', lgbm)]
    estimators2 = [('extra', extra), ('gbdt', gbdt), ('xgb', xgb), ('xgbdart', xgbdart), ('lgbm', lgbm)]
    weights1 = [1, 4, 4, 2]
    weights2 = [1, 1, 4, 4, 2]

    parameters = [{'estimators': [estimators1], 'weights': [weights1]},
                  {'estimators': [estimators2], 'weights': [weights2]}]

    voting = VotingRegressor(estimators=estimators1, n_jobs=5)
    voting_cv = GridSearchCV(voting, parameters, make_scorer(rmsle_score, greater_is_better=False), cv=5, n_jobs=5)
    voting_cv.fit(x_train.values, y_train.values)
    y_pred = voting_cv.predict(x_test.values)

    return y_pred, voting_cv.best_params_


def cyclic_encoding(x, df_test):
    x_new = pd.DataFrame()
    df_test_new = pd.DataFrame()
    x_new['season_sin'] = np.sin(x['season'].values*(2.*np.pi/4))
    x_new['season_cos'] = np.cos(x['season'].values*(2.*np.pi/4))
    x_new['year'] = x['year']
    x_new['month_sin'] = np.sin(x['month'].values*(2.*np.pi/12))
    x_new['month_cos'] = np.cos(x['month'].values*(2.*np.pi/12))
    x_new['hour_sin'] = np.sin(x['hour'].values*(2.*np.pi/24))
    x_new['hour_cos'] = np.cos(x['hour'].values*(2.*np.pi/24))
    x_new['holiday'] = x['holiday']
    x_new['weekday_sin'] = np.sin(x['weekday'].values*(2.*np.pi/7))
    x_new['weekday_cos'] = np.cos(x['weekday'].values*(2.*np.pi/7))
    x_new['workingday'] = x['workingday']
    x_new['weather'] = x['weather']
    x_new['temp'] = x['temp']
    x_new['humidity'] = x['humidity']

    df_test_new['season_sin'] = np.sin(df_test['season'].values*(2.*np.pi/4))
    df_test_new['season_cos'] = np.cos(df_test['season'].values*(2.*np.pi/4))
    df_test_new['year'] = x['year']
    df_test_new['month_sin'] = np.sin(df_test['month'].values*(2.*np.pi/12))
    df_test_new['month_cos'] = np.cos(df_test['month'].values*(2.*np.pi/12))
    df_test_new['hour_sin'] = np.sin(df_test['hour'].values*(2.*np.pi/24))
    df_test_new['hour_cos'] = np.cos(df_test['hour'].values*(2.*np.pi/24))
    df_test_new['holiday'] = df_test['holiday']
    df_test_new['weekday_sin'] = np.sin(df_test['weekday'].values*(2.*np.pi/7))
    df_test_new['weekday_cos'] = np.cos(df_test['weekday'].values*(2.*np.pi/7))
    df_test_new['workingday'] = df_test['workingday']
    df_test_new['weather'] = df_test['weather']
    df_test_new['temp'] = df_test['temp']
    df_test_new['humidity'] = df_test['humidity']

    return x_new, df_test_new
