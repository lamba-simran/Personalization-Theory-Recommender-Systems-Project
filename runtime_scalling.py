import os
from surprise import Reader, Dataset, GridSearch, SVD, SVDpp, NMF, BaselineOnly, KNNWithZScore, KNNWithMeans, KNNBasic
import time
import matplotlib.pyplot as plt


def load_data():
    '''
    Returns a list of data of different sizes
    '''

    #for sample_700
    file_path = os.path.expanduser('data/sample_700.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data0 = Dataset.load_from_file(file_path, reader=reader)

    #for sample_1400
    file_path = os.path.expanduser('data/sample_1400.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data1 = Dataset.load_from_file(file_path, reader=reader)

    #for sample_2100
    file_path = os.path.expanduser('data/sample_2100.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data2 = Dataset.load_from_file(file_path, reader=reader)

    #for sample_full
    file_path = os.path.expanduser('data/sampled_data.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data3 = Dataset.load_from_file(file_path, reader=reader)

    data = [data0, data1, data2, data3]
    return data


def svd_running_time(data):
    '''
        Calculates the running times for training and predictions for SVD

        Args:
            data(Dataset): a list of datasets with different numbers of users

        Returns:
            elapsedtime_SVDtrain: running time for training
            elapsedtime_SVDtest: running time for predictions on testset
    '''
    elapsedtime_SVDtrain = []
    elapsedtime_SVDtest = []

    # tune the parameters on the entire data
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    n_factors = param['n_factors']
    n_epochs = param['n_epochs']

    # using the tuned parameters calculate running times
    for i in range(len(data)):
        # training running time
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        svd = SVD(n_factors=n_factors, n_epochs=n_epochs)
        svd.train(training)
        elapsedtime_SVDtrain.append(time.time() - training_start)

        # prediction running time
        test_start = time.time()
        svd.test(testing)
        elapsedtime_SVDtest.append(time.time() - test_start)
    return elapsedtime_SVDtrain, elapsedtime_SVDtest


def svdpp_running_time(data):
    '''
        Calculates the running times for training and predictions for SVD++

        Args:
            data(Dataset): a list of datasets with different numbers of users

        Returns:
            elapsedtime_SVDpptrain: running time for training
            elapsedtime_SVDpptest: running time for predictions on testset
    '''
    elapsedtime_SVDpptrain = []
    elapsedtime_SVDpptest = []

    # tune the parameters on the entire data
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    n_factors = param['n_factors']
    n_epochs = param['n_epochs']

    # using the tuned parameters calculate running times
    for i in range(len(data)):
        # training running time
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        svdpp = SVDpp(n_factors=n_factors, n_epochs=n_epochs)
        svdpp.train(training)
        elapsedtime_SVDpptrain.append(time.time() - training_start)

        # prediction running time
        test_start = time.time()
        svdpp.test(testing)
        elapsedtime_SVDpptest.append(time.time() - test_start)
    return elapsedtime_SVDpptrain, elapsedtime_SVDpptest


def nmf_running_time(data):
    '''
        Calculates the running times for training and predictions for NMF

        Args:
            data(Dataset): a list of datasets with different numbers of users

        Returns:
            elapsedtime_NMFtrain: running time for training
            elapsedtime_NMFtest: running time for predictions on testset
    '''
    elapsedtime_NMFtrain = []
    elapsedtime_NMFtest = []

    # tune the parameters on the entire data
    param_grid = {'n_factors': [45, 50, 55, 60], 'n_epochs': [45, 50, 55]}
    grid_search = GridSearch(NMF, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    n_factors = param['n_factors']
    n_epochs = param['n_epochs']

    # using the tuned parameters calculate running times
    for i in range(len(data)):
        # training running time
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        nmf = NMF(n_factors=n_factors, n_epochs=n_epochs)
        nmf.train(training)
        elapsedtime_NMFtrain.append(time.time() - training_start)

        # prediction running time
        test_start = time.time()
        nmf.test(testing)
        elapsedtime_NMFtest.append(time.time() - test_start)
    return elapsedtime_NMFtrain, elapsedtime_NMFtest


def base_running_time(data):
    '''
        Calculates the running times for training and predictions for Baseline algorithm

        Args:
            data(Dataset): a list of datasets with different numbers of users

        Returns:
            elapsedtime_Basetrain: running time for training
            elapsedtime_Basetest: running time for predictions on testset
    '''
    elapsedtime_Basetrain = []
    elapsedtime_Basetest = []

    # calculate running times
    for i in range(len(data)):
        # training running time
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        baseline = BaselineOnly()
        baseline.train(training)
        elapsedtime_Basetrain.append(time.time() - training_start)

        # prediction running time
        test_start = time.time()
        baseline.test(testing)
        elapsedtime_Basetest.append(time.time() - test_start)
    return elapsedtime_Basetrain, elapsedtime_Basetrain


def knn_running_time(data):
    '''
        Calculates the running times for training and predictions for Basic KNN

        Args:
            data(Dataset): a list of datasets with different numbers of users

        Returns:
            elapsedtime_KnnBasictrain: running time for training
            elapsedtime_KnnBasictest: running time for predictions on testset
    '''
    elapsedtime_KnnBasictrain = []
    elapsedtime_KnnBasictest = []

    # tune the parameters on the entire data
    param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                    'min_support': [1, 5], 'user_based': [False]}}
    grid_search = GridSearch(KNNBasic, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    k = param['k']
    sim = param['sim_options']['name']
    min_support = param['sim_options']['min_support']
    user_based = param['sim_options']['user_based']

    # using the tuned parameters calculate running times
    for i in range(len(data)):
        # training running time
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        knn = KNNBasic(k=k, name=sim, min_support=min_support, user_based=user_based)
        knn.train(training)
        elapsedtime_KnnBasictrain.append(time.time() - training_start)

        # prediction running time
        test_start = time.time()
        knn.test(testing)
        elapsedtime_KnnBasictest.append(time.time() - test_start)
    return elapsedtime_KnnBasictrain, elapsedtime_KnnBasictest


def knnm_running_time(data):
    '''
        Calculates the running times for training and predictions for KNN with Means

        Args:
            data(Dataset): a list of datasets with different numbers of users

        Returns:
            elapsedtime_KnnMeanstrain: running time for training
            elapsedtime_KnnMeanstest: running time for predictions on testset
    '''
    elapsedtime_KnnMeanstrain = []
    elapsedtime_KnnMeanstest = []

    # tune the parameters on the entire data
    param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                    'min_support': [1, 5], 'user_based': [False]}}
    grid_search = GridSearch(KNNWithMeans, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    k = param['k']
    sim = param['sim_options']['name']
    min_support = param['sim_options']['min_support']
    user_based = param['sim_options']['user_based']

    # using the tuned parameters calculate running times
    for i in range(len(data)):
        # training running time
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        knnm = KNNWithMeans(k=k, name=sim, min_support=min_support, user_based=user_based)
        knnm.train(training)
        elapsedtime_KnnMeanstrain.append(time.time() - training_start)

        # prediction running time
        test_start = time.time()
        knnm.test(testing)
        elapsedtime_KnnMeanstest.append(time.time() - test_start)
    return elapsedtime_KnnMeanstrain, elapsedtime_KnnMeanstest


def knnz_running_time(data):
    '''
        Calculates the running times for training and predictions for KNN with Z-score

        Args:
            data(Dataset): a list of datasets with different numbers of users

        Returns:
            elapsedtime_KnnZtrain: running time for training
            elapsedtime_KnnZtest: running time for predictions on testset
    '''
    elapsedtime_KnnZtrain = []
    elapsedtime_KnnZtest = []

    # tune the parameters on the entire data
    param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                    'min_support': [1, 5], 'user_based': [False]}}
    grid_search = GridSearch(KNNWithZScore, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    k = param['k']
    sim = param['sim_options']['name']
    min_support = param['sim_options']['min_support']
    user_based = param['sim_options']['user_based']

    # using the tuned parameters calculate running times
    for i in range(len(data)):
        # training running time
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        knnz = KNNWithZScore(k=k, name=sim, min_support=min_support, user_based=user_based)
        knnz.train(training)
        elapsedtime_KnnZtrain.append(time.time() - training_start)

        # prediction running time
        test_start = time.time()
        knnz.test(testing)
        elapsedtime_KnnZtest.append(time.time() - test_start)
    return elapsedtime_KnnZtrain, elapsedtime_KnnZtest


def main():
    # load data of different sizes
    data = load_data()

    # store the running times for training and test for each algorithm
    elapsedtime_SVDtrain, elapsedtime_SVDtest = svd_running_time(data)
    elapsedtime_SVDpptrain, elapsedtime_SVDpptest = svdpp_running_time(data)
    elapsedtime_NMFtrain, elapsedtime_NMFtest = nmf_running_time(data)
    elapsedtime_Basetrain, elapsedtime_Basetest = base_running_time(data)
    elapsedtime_KnnBasictrain, elapsedtime_KnnBasictest = knn_running_time(data)
    elapsedtime_KnnMeanstrain, elapsedtime_KnnMeanstest = knnm_running_time(data)
    elapsedtime_KnnZtrain, elapsedtime_KnnZtest = knnz_running_time(data)

    # plot training running time
    users_N = [700, 1400, 2100, 3000]
    plt.plot(users_N, elapsedtime_Basetrain, 'r', label="BaselineOnly")
    plt.plot(users_N, elapsedtime_KnnBasictrain, 'orange', label="KNNBasic")
    plt.plot(users_N, elapsedtime_KnnMeanstrain, 'skyblue', label="KNNMean")
    plt.plot(users_N, elapsedtime_KnnZtrain, 'g', label="KNNZScore")
    plt.plot(users_N, elapsedtime_SVDtrain, 'b', label="SVD")
    plt.plot(users_N, elapsedtime_SVDpptrain, 'y', label="SVDpp")
    plt.plot(users_N, elapsedtime_NMFtrain, 'm', label="NMF")

    plt.ylabel('Training Running Time')
    plt.xlabel('Input Size(Users)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    # plot test running time
    plt.plot(users_N, elapsedtime_Basetest, 'r', label="BaselineOnly")
    plt.plot(users_N, elapsedtime_KnnBasictest, 'orange', label="KNNBasic")
    plt.plot(users_N, elapsedtime_KnnMeanstest, 'skyblue', label="KNNMean")
    plt.plot(users_N, elapsedtime_KnnZtrain, 'g', label="KNNZScore")
    plt.plot(users_N, elapsedtime_SVDtest, 'b', label="SVD")
    plt.plot(users_N, elapsedtime_SVDpptest, 'y', label="SVDpp")
    plt.plot(users_N, elapsedtime_NMFtest, 'm', label="NMF")

    plt.ylabel('Testing Running Time')
    plt.xlabel('Input Size(Users)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
    main()