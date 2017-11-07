import os
from surprise import Reader, Dataset, GridSearch, SVD, SVDpp, NMF, accuracy, BaselineOnly, KNNWithZScore, KNNWithMeans, KNNBasic
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def get_top_n(predictions, n=10):
    '''
    modified from https://pypkg.com/pypi/scikit-surprise/f/examples/top_n_recommendations.py

    Return the number of unique predictions of top-N recommendation for each user.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
        the number of unique predictions
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
    # recommended_items stores items recommended to users in test data
    length = len(set(recommended_items))
    return length


def svdpp(data, training, testing):
    '''
    Tune SVD++ parameters then calculates RMSE, coverage and running time of SVD++

    Args:
        data(Dataset): the whole dataset divided into 5 folds
        training(Dataset): training dataset
        testing(Dataset): test dataset

    Returns:
        rmse: RMSE of SVD++ with optimized parameters
        top_n: number of unique predictions for top n items
    '''
    # candidate parameters
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}

    # optimize parameters
    grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('SVDpp:', param)

    # fit model using the optimized parameters
    svdpp = SVDpp(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    svdpp.train(training)

    # evaluate the model using test data
    predictions = svdpp.test(testing)
    top_n = get_top_n(predictions, n=5)
    rmse = accuracy.rmse(predictions, verbose=True)

    return rmse, top_n


def nmf(data, training, testing):
    '''
    Tune NMF parameters then calculates RMSE, coverage and running time of NMF

    Args:
        data(Dataset): the whole dataset divided into 5 folds
        training(Dataset): training dataset
        testing(Dataset): test dataset

    Returns:
        rmse: RMSE of NMF with optimized parameters
        top_n: number of unique predictions for top n items
    '''

    # candidate parameters
    nmf_param_grid = {'n_factors': [45, 50, 55, 60], 'n_epochs': [45, 50, 55]}

    # optimize parameters
    grid_search = GridSearch(NMF, nmf_param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('NMF:', param)

    # fit model using the optimized parameters
    nmf = NMF(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    nmf.train(training)

    # evaluate the model using test data
    predictions = nmf.test(testing)
    top_n = get_top_n(predictions, n=5)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, top_n


def baseline(training, testing):
    '''
    Calculates RMSE, coverage and running time of Baseline model

    Args:
        training(Dataset): training dataset
        testing(Dataset): test dataset

    Returns:
        rmse: RMSE of Baseline with optimized parameters
        top_n: number of unique predictions for top n items
    '''

    # fit model
    baseline = BaselineOnly()
    baseline.train(training)

    # evaluate the model using test data
    predictions = baseline.test(testing)
    top_n = get_top_n(predictions, n=5)
    rmse = accuracy.rmse(predictions, verbose=True)

    return rmse, top_n


def knn_z(data, training, testing):
    '''
    Tune KNN with Z-score parameters then calculates RMSE, coverage and running time of KNN with Z-score

    Args:
        data(Dataset): the whole dataset divided into 5 folds
        training(Dataset): training dataset
        testing(Dataset): test dataset

    Returns:
        rmse: RMSE of KNN with Z-score with optimized parameters
        top_n: number of unique predictions for top n items
    '''

    # candidate parameters
    knn_param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                        'min_support': [1, 5],'user_based': [False]}}

    # optimize parameters
    knnz_grid_search = GridSearch(KNNWithZScore, knn_param_grid, measures=['RMSE'], verbose=False)
    knnz_grid_search.evaluate(data)
    param = knnz_grid_search.best_params['RMSE']
    print('KNNWithZScore:', param)

    # fit model using the optimized parameters
    knnz = KNNWithZScore(k = param['k'], name=param['sim_options']['name'],
                         min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'])
    knnz.train(training)

    # evaluate the model using test data
    predictions = knnz.test(testing)
    rmse = accuracy.rmse(predictions, verbose=True)
    top_n = get_top_n(predictions, n=5)

    return rmse, top_n


def knn(data, training, testing):
    '''
        Tune Basic KNN parameters then calculates RMSE, coverage and running time of Basic KNN

        Args:
            data(Dataset): the whole dataset divided into 5 folds
            training(Dataset): training dataset
            testing(Dataset): test dataset

        Returns:
            rmse: RMSE of Basic KNN with optimized parameters
            top_n: number of unique predictions for top n items
    '''

    # candidate parameters
    knn_param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                        'min_support': [1, 5], 'user_based': [False]}}

    # optimize parameters
    knn_grid_search = GridSearch(KNNBasic, knn_param_grid, measures=['RMSE'], verbose=False)
    knn_grid_search.evaluate(data)
    param = knn_grid_search.best_params['RMSE']
    print('KNNBasic:', param)

    # fit model using the optimized parameters
    knn = KNNBasic(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
    knn.train(training)

    # evaluate the model using test data
    predictions = knn.test(testing)
    top_n = get_top_n(predictions, n=5)

    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, top_n


def knn_m(data, training, testing):
    '''
        Tune KNN with Means parameters then calculates RMSE, coverage and running time of KNN with Means

        Args:
            data(Dataset): the whole dataset divided into 5 folds
            training(Dataset): training dataset
            testing(Dataset): test dataset

        Returns:
            rmse: RMSE of KNN with Means with optimized parameters
            top_n: number of unique predictions for top n items
    '''

    # candidate parameters
    knn_param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                        'min_support': [1, 5], 'user_based': [False]}}

    # optimize parameters
    knnm_grid_search = GridSearch(KNNWithMeans, knn_param_grid, measures=['RMSE'], verbose=False)
    knnm_grid_search.evaluate(data)
    param = knnm_grid_search.best_params['RMSE']
    print('KNNWithMeans:', param)

    # fit model using the optimized parameters
    knnm = KNNWithMeans(k=param['k'], name=param['sim_options']['name'],
                        min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'])
    knnm.train(training)

    # evaluate the model using test data
    predictions = knnm.test(testing)
    top_n = get_top_n(predictions, n=5)
    rmse = accuracy.rmse(predictions, verbose=True)

    return rmse, top_n


def svd(data, training, testing):
    '''
        Tune SVD parameters then calculates RMSE, coverage and running time of SVD

        Args:
            data(Dataset): the whole dataset divided into 5 folds
            training(Dataset): training dataset
            testing(Dataset): test dataset

        Returns:
            rmse: RMSE of SVD with Z-score with optimized parameters
            top_n: number of unique predictions for top n items
    '''

    # candidate parameters
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}

    # optimize parameters
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    param = grid_search.best_params['RMSE']
    print('SVD:', param)

    # fit model using the optimized parameters
    svd = SVD(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
    svd.train(training)

    # evaluate the model using test data
    predictions = svd.test(testing)
    top_n = get_top_n(predictions, n=5)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, top_n


if __name__ == "__main__":
    # import data and divide it into 5-folds
    file_path = os.path.expanduser('sample_700 copy.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)  # data can now be used normally

    # create training and test data
    training = data.build_full_trainset()
    testing = training.build_anti_testset()

    # using created datasets, tune parameters, fit models and evaluate them
    svd_rmse, svd_cov = svd(data, training, testing)
    svdpp_rmse, svdpp_cov = svdpp(data, training, testing)
    nmf_rmse, nmf_cov = nmf(data, training, testing)
    baseline_rmse, baseline_cov = baseline(training, testing)
    knnz_rmse, knnz_cov = knn_z(data, training, testing)
    knn_rmse, knn_cov = knn(data, training, testing)
    knnm_rmse, knnm_cov = knn_m(data, training, testing)

    # plot accuracy of the models
    objects = ('SVD', 'SVD++', 'NMF', 'Baseline', 'KNNWithZScore', 'KNNBasic', 'KNNWithMeans')
    y_pos = np.arange(len(objects))
    performance = [svd_rmse, svdpp_rmse, nmf_rmse, baseline_rmse, knnz_rmse, knn_rmse, knnm_rmse]
    plt.bar(y_pos, performance, align='center', alpha=0.1)
    plt.xticks(y_pos, objects)
    plt.show()

    # plot coverage of the models
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
# 60 30 48 7 33 47 34
