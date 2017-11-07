import os
from surprise import Reader, Dataset, GridSearch, SVD, SVDpp, NMF, accuracy, BaselineOnly, KNNWithZScore, KNNWithMeans, KNNBasic
import numpy as np
import matplotlib.pyplot as plt


def svdpp_rmse(data, training, testing):
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('SVDpp:', param)
    svdpp = SVDpp(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    svdpp.train(training)
    predictions = svdpp.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)

    return rmse


def nmf_rmse(data, training, testing):
    nmf_param_grid = {'n_factors': [45, 50, 55, 60], 'n_epochs': [45, 50, 55]}
    grid_search = GridSearch(NMF, nmf_param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('NMF:', param)
    nmf = NMF(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    nmf.train(training)
    predictions = nmf.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse


def baseline_rmse(training, testing):
    baseline = BaselineOnly()
    baseline.train(training)
    predictions = baseline.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse


def knnz_rmse(data, training, testing):
    knn_param_grid = {'k': [20,30, 40], 'sim_options': {'name': ['msd', 'cosine', 'pearson'], 'min_support': [5,10],
                                                        'user_based': [False]}}

    knnz_grid_search = GridSearch(KNNWithZScore, knn_param_grid, measures=['RMSE'], verbose=False)
    knnz_grid_search.evaluate(data)
    param = knnz_grid_search.best_params['RMSE']
    print('KNNWithZScore:', param)
    knnz = KNNWithZScore(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
    knnz.train(training)
    predictions = knnz.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse


def knn_rmse(data, training, testing):
    knn_param_grid = {'k': [20, 30, 40], 'sim_options': {'name': ['msd', 'cosine', 'pearson'], 'min_support': [5, 10],
                                                         'user_based': [False]}}
    knn_grid_search = GridSearch(KNNBasic, knn_param_grid, measures=['RMSE'], verbose=False)
    knn_grid_search.evaluate(data)
    param = knn_grid_search.best_params['RMSE']
    print('KNNBasic:', param)
    knn = KNNBasic(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
    knn.train(training)
    predictions = knn.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse


def knnm_rmse(data, training, testing):
    knn_param_grid = {'k': [20,30, 40], 'sim_options': {'name': ['msd', 'cosine', 'pearson'], 'min_support': [5,10],
                                                        'user_based': [False]}}
    knnm_grid_search = GridSearch(KNNWithMeans, knn_param_grid, measures=['RMSE'], verbose=False)
    knnm_grid_search.evaluate(data)
    param = knnm_grid_search.best_params['RMSE']
    print('KNNWithMeans:', param)
    knnm = KNNWithMeans(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
    knnm.train(training)
    predictions = knnm.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse


def svd_rmse(data, training, testing):
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('SVD:', param)
    svd = SVD(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    svd.train(training)
    predictions = svd.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse



file_path = os.path.expanduser('sampled_data.csv')
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)  # data can now be used normally
training = data.build_full_trainset()
testing = training.build_anti_testset()
svd_rmse = svd_rmse(data, training, testing)
svdpp_rmse = svdpp_rmse(data, training, testing)
nmf_rmse = nmf_rmse(data, training, testing)
baseline_rmse = baseline_rmse(training, testing)
knnz_rmse = knnz_rmse(data, training, testing)
knn_rmse = knn_rmse(data, training, testing)
knnm_rmse = knnm_rmse(data, training, testing)

objects = ("SVD", "SVD++", "NMF", "Baseline", 'KNNWithZScore', 'KNNBasic', 'KNNWithMeans')
y_pos = np.arange(len(objects))
performance = [svd_rmse, svdpp_rmse, nmf_rmse, baseline_rmse, knnz_rmse, knn_rmse, knnm_rmse]
plt.bar(y_pos, performance, align='center', alpha=0.1)
plt.xticks(y_pos, objects)
plt.show()

# SVD: {'n_factors': 25, 'n_epochs': 40}
# SVDpp: {'n_factors': 25, 'n_epochs': 10}
# NMF: {'n_factors': 45, 'n_epochs': 50}
# KNNWithZScore: {'k': 20, 'sim_options': {'name': 'cosine', 'min_support': 5, 'user_based': False}}
# KNNBasic: {'k': 20, 'sim_options': {'name': 'msd', 'min_support': 10, 'user_based': False}}
# KNNWithMeans: {'k': 20, 'sim_options': {'name': 'pearson', 'min_support': 5, 'user_based': False}}