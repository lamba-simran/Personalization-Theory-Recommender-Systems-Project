import os
import surprise
from surprise import Reader, Dataset, GridSearch, accuracy,  SVD, SVDpp, NMF, BaselineOnly, evaluate, KNNWithZScore, KNNWithMeans, KNNBasic
import numpy as np
import time

def load_data():
    #for sample_700
    file_path = os.path.expanduser('/Users/lamba_s/Desktop/personalization-theory-master/sample_700.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data0 = Dataset.load_from_file(file_path, reader=reader)

    #for sample_1400
    file_path = os.path.expanduser('/Users/lamba_s/Desktop/personalization-theory-master/sample_1400.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data1 = Dataset.load_from_file(file_path, reader=reader)

    #for sample_2100
    file_path = os.path.expanduser('/Users/lamba_s/Desktop/personalization-theory-master/sample_2100.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data2 = Dataset.load_from_file(file_path, reader=reader)

    #for sample_full
    file_path = os.path.expanduser('/Users/lamba_s/Desktop/personalization-theory-master/sampled_data.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data3 = Dataset.load_from_file(file_path, reader=reader)

    data = [data0, data1, data2, data3]

    return data


def svd_running_time(elapsedtime_SVDtrain, elapsedtime_SVDtest, n_factors, n_epochs, data):
    for i in range(len(data)):
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        svd = SVD(n_factors=n_factors, n_epochs=n_epochs)
        svd.train(training)
        elapsedtime_SVDtrain.append(time.time() - training_start)

        test_start = time.time()
        svd.test(testing)
        elapsedtime_SVDtest.append(time.time() - test_start)

    return elapsedtime_SVDtrain, elapsedtime_SVDtest


def svdpp_running_time(elapsedtime_SVDpptrain, elapsedtime_SVDpptest, n_factors, n_epochs, data):
    for i in range(len(data)):
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        svdpp = SVDpp(n_factors=n_factors, n_epochs=n_epochs)
        svdpp.train(training)
        elapsedtime_SVDpptrain.append(time.time() - training_start)

        test_start = time.time()
        svdpp.test(testing)
        elapsedtime_SVDpptest.append(time.time() - test_start)

    return elapsedtime_SVDpptrain, elapsedtime_SVDpptest




if __name__ == "__main__":
    data = load_data()
    elapsedtime_SVDtrain = []
    elapsedtime_SVDtest = []
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']

    elapsedtime_SVD = svd_running_time(elapsedtime_SVDtrain, elapsedtime_SVDtest, param['n_factors'],
                                       param['n_epochs'], data)
    print(elapsedtime_SVD)

    elapsedtime_SVDpptrain = []
    elapsedtime_SVDpptest = []
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']

    elapsedtime_SVDpp = svdpp_running_time(elapsedtime_SVDpptrain, elapsedtime_SVDpptest, param['n_factors'],
                                     param['n_epochs'], data)
    print(elapsedtime_SVDpp)