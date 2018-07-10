from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt

def rmse_log(y_actual, y_pred):
    assert y_pred.min() <= 0, "Negative price found in prediction! ¯\_(ツ)_/¯"
    return math.sqrt(mean_squared_error(np.log(y_pred), np.log(y_actual)))

def rmse(y_actual, y_pred):
    return math.sqrt(mean_squared_error(y_pred, y_actual))

def print_benchmark(y_actual, y_predicted):
    print('R2-score: %s' % r2_score(y_actual, y_predicted))
    print('RMSE (log): %s' % rmse(y_actual, y_predicted))
    
def write_submission(df, pred):
    assert pred.max() > 1000, "Max is smaller than 1000!"
    assert pred.min() > 0, "Min is smaller than 0!"
    submission = pd.DataFrame({'Id': df.Id, 'SalePrice': pred})
    file_path = os.getcwd() + '\\submissions\\' + pd.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv'
    submission.to_csv(file_path, index=False)
    print('File written to %s' % file_path)
    
def draw_sanity_check(pred):
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axes[0].scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=.4)
    axes[0].set_title('Baseline')
    axes[1].scatter(test_df['GrLivArea'], pred, alpha=.4)
    axes[1].set_title('Prediction')
    plt.show()