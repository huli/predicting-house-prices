from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import math

def rmse_log(y_actual, y_pred):
    assert y_pred.min() >= 0, "Negative price found in prediction! ¯\_(ツ)_/¯"
    return math.sqrt(mean_squared_error(np.log(y_pred), np.log(y_actual)))

def print_benchmark(y_actual, y_predicted):
    print('R2-score: %s' % r2_score(y_actual, y_predicted))
    print('RMSE (log): %s' % rmse_log(y_actual, y_predicted))