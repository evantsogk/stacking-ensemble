import matplotlib.pyplot as plt
import utils
from stacking_ensemble import StackingEnsemble

_author_ = "Evangelos Tsogkas p3150185"

x_train, y_train_casual, y_train_registered, x_test = utils.load_data('dataset/train.csv', 'dataset/test.csv')
x_train_enc, x_test_enc = utils.one_hot_encode(x_train, x_test)

model = StackingEnsemble()
model.fit(x_train.values, x_train_enc, y_train_casual.values, y_train_registered.values, x_test.values, x_test_enc)
y_pred = model.predict()

utils.export_submission(y_pred, 'submission/final_submission.csv')

plt.show()
