import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow import keras

_author_ = "Evangelos Tsogkas p3150185"


class StackingEnsemble:
    """A stacking ensemble specifically for the bike sharing demand prediction.

    It consists of six base regressors and one meta regressor. The meta regressor is a Neural Network and the base
    regressors are the following: Extra Trees, Gradient Boosting, XGBoost, XGBoost trained with dart, LightGBM, Neural
    Network.

    It follows the KFold prediction approach and the base regressors predict the 'casual' and 'registered' values
    separately, which are then added together to get the wanted 'count' values.
    """

    def _init_(self):
        self.meta_train_set = None
        self.meta_test_set = None
        self.meta_regressor = None

    def _fit_base_regressors(self, x, x_enc, y_train_casual, y_train_registered, x_test, x_test_enc):
        """The base regressors are trained to create the meta-train and meta-test sets based on their predictions.
        """

        # Base Regressors
        extra = ExtraTreesRegressor(n_jobs=-1, random_state=0, n_estimators=500, max_depth=26)
        gbdt = GradientBoostingRegressor(random_state=0, tol=1e-6, loss='lad', n_estimators=80, subsample=0.8,
                                         max_depth=13)
        xgb = XGBRegressor(n_jobs=-1, random_state=0, objective='reg:squarederror', n_estimators=60, subsample=0.7,
                           max_depth=15, reg_lambda=6.5)
        xgbdart = XGBRegressor(n_jobs=-1, random_state=0, objective='reg:squarederror', booster='dart', n_estimators=50,
                               subsample=0.7, max_depth=17, reg_lambda=6)
        lgbm = LGBMRegressor(random_state=0, n_jobs=-1, min_child_samples=0, boosting_type='dart', n_estimators=350,
                             subsample=0.1, num_leaves=550, reg_lambda=6)
        regularizer = keras.regularizers.l2(0.0001)
        nn = keras.Sequential([
            keras.layers.Dense(400, activation='relu', input_dim=x_enc.shape[1], kernel_regularizer=regularizer),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(200, activation='relu', kernel_regularizer=regularizer),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizer),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='linear')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        nn.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['msle'])
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)

        # Create meta-train set

        extra_train, gbdt_train, xgb_train, xgbdart_train, lgbm_train, nn_train = [], [], [], [], [], []
        kf = KFold(n_splits=5)

        for train_index, test_index in kf.split(x):
            x_train, x_val = x[train_index], x[test_index]
            x_train_enc, x_val_enc = x_enc[train_index], x_enc[test_index]
            y_casual = y_train_casual[train_index]
            y_registered = y_train_registered[train_index]

            # Extra Trees
            extra.fit(x_train, y_casual)
            y_pred_casual = extra.predict(x_val)
            extra.fit(x_train, y_registered)
            y_pred_registered = extra.predict(x_val)
            extra_train.extend(np.add(y_pred_casual, y_pred_registered))

            # Gradient Boosting
            gbdt.fit(x_train, y_casual)
            y_pred_casual = gbdt.predict(x_val)
            gbdt.fit(x_train, y_registered)
            y_pred_registered = gbdt.predict(x_val)
            gbdt_train.extend(np.add(y_pred_casual, y_pred_registered))

            # XGBoost
            xgb.fit(x_train, y_casual)
            y_pred_casual = xgb.predict(x_val)
            xgb.fit(x_train, y_registered)
            y_pred_registered = xgb.predict(x_val)
            xgb_train.extend(np.add(y_pred_casual, y_pred_registered))

            # XGBoost Dart
            xgbdart.fit(x_train, y_casual)
            y_pred_casual = xgbdart.predict(x_val)
            xgbdart.fit(x_train, y_registered)
            y_pred_registered = xgbdart.predict(x_val)
            xgbdart_train.extend(np.add(y_pred_casual, y_pred_registered))

            # LightGBM
            lgbm.fit(x_train, y_casual)
            y_pred_casual = lgbm.predict(x_val)
            lgbm.fit(x_train, y_registered)
            y_pred_registered = lgbm.predict(x_val)
            lgbm_train.extend(np.add(y_pred_casual, y_pred_registered))

            # Neural Network
            nn.fit(x_train_enc, y_casual, epochs=500, batch_size=32, verbose=1, validation_split=0.1,
                   callbacks=[early_stopping])
            y_pred_casual = nn.predict(x_val_enc).flatten()
            nn.fit(x_train_enc, y_registered, epochs=500, batch_size=32, verbose=1, validation_split=0.1,
                   callbacks=[early_stopping])
            y_pred_registered = nn.predict(x_val_enc).flatten()
            nn_train.extend(np.add(y_pred_casual, y_pred_registered))

        self.meta_train_set = np.concatenate(([extra_train], [gbdt_train], [xgb_train], [xgbdart_train], [lgbm_train],
                                              [nn_train]), axis=0).T

        # Create meta-test set

        # Extra Trees
        extra.fit(x, y_train_casual)
        y_pred_casual = extra.predict(x_test)
        extra.fit(x, y_train_registered)
        y_pred_registered = extra.predict(x_test)
        extra_test = np.add(y_pred_casual, y_pred_registered)

        # Gradient Boosting
        gbdt.fit(x, y_train_casual)
        y_pred_casual = gbdt.predict(x_test)
        gbdt.fit(x, y_train_registered)
        y_pred_registered = gbdt.predict(x_test)
        gbdt_test = np.add(y_pred_casual, y_pred_registered)

        # XGBoost
        xgb.fit(x, y_train_casual)
        y_pred_casual = xgb.predict(x_test)
        xgb.fit(x, y_train_registered)
        y_pred_registered = xgb.predict(x_test)
        xgb_test = np.add(y_pred_casual, y_pred_registered)

        # XGBoost Dart
        xgbdart.fit(x, y_train_casual)
        y_pred_casual = xgbdart.predict(x_test)
        xgbdart.fit(x, y_train_registered)
        y_pred_registered = xgbdart.predict(x_test)
        xgbdart_test = np.add(y_pred_casual, y_pred_registered)

        # LightGBM
        lgbm.fit(x, y_train_casual)
        y_pred_casual = lgbm.predict(x_test)
        lgbm.fit(x, y_train_registered)
        y_pred_registered = lgbm.predict(x_test)
        lgbm_test = np.add(y_pred_casual, y_pred_registered)

        # Neural Network
        nn.fit(x_enc, y_train_casual, epochs=500, batch_size=32, verbose=1, validation_split=0.1,
               callbacks=[early_stopping])
        y_pred_casual = nn.predict(x_test_enc).flatten()
        nn.fit(x_enc, y_train_registered, epochs=500, batch_size=32, verbose=1, validation_split=0.1,
               callbacks=[early_stopping])
        y_pred_registered = nn.predict(x_test_enc).flatten()
        nn_test = np.add(y_pred_casual, y_pred_registered)

        self.meta_test_set = np.concatenate(([extra_test], [gbdt_test], [xgb_test], [xgbdart_test], [lgbm_test],
                                             [nn_test]), axis=0).T

    def _fit_meta_regressor(self, y_train_casual, y_train_registered):
        """The Neural Network meta-regressor is trained with the meta-train set created from the predictions of the
        base regressors.
        """
        y_train = np.add(y_train_casual, y_train_registered)
        regularizer = keras.regularizers.l2(0.0001)
        self.meta_regressor = keras.Sequential([
            keras.layers.Dense(100, activation='relu', input_dim=self.meta_train_set.shape[1],
                               kernel_regularizer=regularizer),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='linear')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.meta_regressor.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['msle'])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = self.meta_regressor.fit(self.meta_train_set, y_train, epochs=500, batch_size=8, verbose=1,
                                          validation_split=0.1, callbacks=[early_stopping])
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')

    def fit(self, x, x_enc, y_train_casual, y_train_registered, x_test, x_test_enc):
        """
        Trains the base regressors and the meta-regressor.
        """
        self._fit_base_regressors(x, x_enc, y_train_casual, y_train_registered, x_test, x_test_enc)
        self._fit_meta_regressor(y_train_casual, y_train_registered)

    def predict(self):
        """Returns the predictions of the meta-regressor
        """
        return self.meta_regressor.predict(self.meta_test_set).flatten()
