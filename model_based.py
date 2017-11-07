import os
from surprise import Reader, Dataset, GridSearch, SVD, SVDpp, NMF, accuracy, BaselineOnly, KNNWithZScore, KNNWithMeans, KNNBasic
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    recommended_items = []
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n][0][0]
        for recommendation in user_ratings[:n]:
            recommended_items.append(recommendation[0])
    length = len(set(recommended_items))
    return length


def svdpp_rmse(data, training, testing):
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('SVDpp:', param)
    svdpp = SVDpp(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    svdpp.train(training)
    predictions = svdpp.test(testing)
    top_n = get_top_n(predictions, n=5)
    rmse = accuracy.rmse(predictions, verbose=True)

    return rmse, top_n


def nmf_rmse(data, training, testing):
    nmf_param_grid = {'n_factors': [45, 50, 55, 60], 'n_epochs': [45, 50, 55]}
    grid_search = GridSearch(NMF, nmf_param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('NMF:', param)
    nmf = NMF(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    nmf.train(training)
    predictions = nmf.test(testing)
    top_n = get_top_n(predictions, n=5)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, top_n


def baseline_rmse(training, testing):
    baseline = BaselineOnly()
    baseline.train(training)
    predictions = baseline.test(testing)
    top_n = get_top_n(predictions, n=5)

    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, top_n


def knnz_rmse(data, training, testing):
    knn_param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                        'min_support': [1, 5],'user_based': [False]}}

    knnz_grid_search = GridSearch(KNNWithZScore, knn_param_grid, measures=['RMSE'], verbose=False)
    knnz_grid_search.evaluate(data)
    param = knnz_grid_search.best_params['RMSE']
    print('KNNWithZScore:', param)
    knnz = KNNWithZScore(k = param['k'], name=param['sim_options']['name'],
                         min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'])
    knnz.train(training)
    predictions = knnz.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)
    top_n = get_top_n(predictions, n=5)

    return rmse, top_n


def knn_rmse(data, training, testing):
    knn_param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                        'min_support': [1, 5], 'user_based': [False]}}
    knn_grid_search = GridSearch(KNNBasic, knn_param_grid, measures=['RMSE'], verbose=False)
    knn_grid_search.evaluate(data)
    param = knn_grid_search.best_params['RMSE']
    print('KNNBasic:', param)
    knn = KNNBasic(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
    knn.train(training)
    predictions = knn.test(testing)
    top_n = get_top_n(predictions, n=5)

    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, top_n


def knnm_rmse(data, training, testing):
    knn_param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                        'min_support': [1, 5], 'user_based': [False]}}
    knnm_grid_search = GridSearch(KNNWithMeans, knn_param_grid, measures=['RMSE'], verbose=False)
    knnm_grid_search.evaluate(data)
    param = knnm_grid_search.best_params['RMSE']
    print('KNNWithMeans:', param)
    knnm = KNNWithMeans(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
    knnm.train(training)
    predictions = knnm.test(testing)
    top_n = get_top_n(predictions, n=5)

    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, top_n


def svd_rmse(data, training, testing):
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('SVD:', param)
    svd = SVD(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    svd.train(training)
    predictions = svd.test(testing)
    top_n = get_top_n(predictions, n=5)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, top_n


if __name__ == "__main__":
    file_path = os.path.expanduser('sample_700 copy.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)  # data can now be used normally
    training = data.build_full_trainset()
    testing = training.build_anti_testset()
    svd_rmse, svd_cov = svd_rmse(data, training, testing)
    svdpp_rmse, svdpp_cov = svdpp_rmse(data, training, testing)
    nmf_rmse, nmf_cov = nmf_rmse(data, training, testing)
    baseline_rmse, baseline_cov = baseline_rmse(training, testing)
    knnz_rmse, knnz_cov = knnz_rmse(data, training, testing)
    knn_rmse, knn_cov = knn_rmse(data, training, testing)
    knnm_rmse, knnm_cov = knnm_rmse(data, training, testing)

    objects = ('SVD', 'SVD++', 'NMF', 'Baseline', 'KNNWithZScore', 'KNNBasic', 'KNNWithMeans')
    y_pos = np.arange(len(objects))
    performance = [svd_rmse, svdpp_rmse, nmf_rmse, baseline_rmse, knnz_rmse, knn_rmse, knnm_rmse]
    plt.bar(y_pos, performance, align='center', alpha=0.1)
    plt.xticks(y_pos, objects)
    plt.show()

    objects = ('SVD', 'SVD++', 'NMF', 'Baseline', 'KNNWithZScore', 'KNNBasic', 'KNNWithMeans')
    y_pos = np.arange(len(objects))
    performance = [svd_cov, svdpp_cov, nmf_cov, baseline_cov, knnz_cov, knn_cov, knnm_cov]
    print(svd_cov, svdpp_cov, nmf_cov, baseline_cov, knnz_cov, knn_cov, knnm_cov)
    plt.bar(y_pos, performance, align='center', alpha=0.1)
    plt.xticks(y_pos, objects)
    plt.show()
# SVD: {'n_factors': 25, 'n_epochs': 50}
# RMSE: 0.4666
# SVDpp: {'n_factors': 25, 'n_epochs': 10}
# RMSE: 0.2783
# NMF: {'n_factors': 45, 'n_epochs': 50}
# RMSE: 1.2048
# KNNWithZScore: {'k': 5, 'sim_options': {'name': 'cosine', 'min_support': 5, 'user_based': False}}
# RMSE: 1.2176
# KNNBasic: {'k': 5, 'sim_options': {'name': 'msd', 'min_support': 5, 'user_based': False}}
# RMSE: 0.5045
# KNNWithMeans: {'k': 5, 'sim_options': {'name': 'cosine', 'min_support': 5, 'user_based': False}}
# RMSE: 1.2152
# top 1
# 23 15 24 2 26 33 25
# top 3
# 56 28 44 7 33 47 34
# top 5
# 56 27 44 7 33 47 34
