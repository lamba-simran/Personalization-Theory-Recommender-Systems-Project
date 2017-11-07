import os
from surprise import Reader, Dataset, GridSearch, SVD, SVDpp, NMF, BaselineOnly, evaluate, KNNWithZScore, KNNWithMeans, KNNBasic
import time
import matplotlib.pyplot as plt

def load_data():
    #for sample_700
    print('here')
    file_path = os.path.expanduser('sample_700.csv')
    print(file_path)
    reader = Reader(line_format='user item rating', sep=',')
    data0 = Dataset.load_from_file(file_path, reader=reader)
    print('done')
    #for sample_1400
    file_path = os.path.expanduser('sample_1400.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data1 = Dataset.load_from_file(file_path, reader=reader)

    #for sample_2100
    file_path = os.path.expanduser('sample_2100.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data2 = Dataset.load_from_file(file_path, reader=reader)

    #for sample_full
    file_path = os.path.expanduser('sampled_data.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data3 = Dataset.load_from_file(file_path, reader=reader)

    data = [data0, data1, data2, data3]

    return data


def svd_running_time(elapsedtime_SVDtrain, elapsedtime_SVDtest, data):
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    n_factors = param['n_factors']
    n_epochs = param['n_epochs']

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


def svdpp_running_time(elapsedtime_SVDpptrain, elapsedtime_SVDpptest, data):
    param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    n_factors = param['n_factors']
    n_epochs = param['n_epochs']

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


def nmf_running_time(elapsedtime_NMFtrain, elapsedtime_NMFtest, data):
    param_grid = {'n_factors': [45, 50, 55, 60], 'n_epochs': [45, 50, 55]}
    grid_search = GridSearch(NMF, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    n_factors = param['n_factors']
    n_epochs = param['n_epochs']

    for i in range(len(data)):
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        nmf = NMF(n_factors=n_factors, n_epochs=n_epochs)
        nmf.train(training)
        elapsedtime_NMFtrain.append(time.time() - training_start)

        test_start = time.time()
        nmf.test(testing)
        elapsedtime_NMFtest.append(time.time() - test_start)


def base_running_time(elapsedtime_Basetrain, elapsedtime_Basetest, data):
    for i in range(len(data)):
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        baseline = BaselineOnly()
        baseline.train(training)
        elapsedtime_Basetrain.append(time.time() - training_start)

        test_start = time.time()
        baseline.test(testing)
        elapsedtime_Basetest.append(time.time() - test_start)


def knn_running_time(elapsedtime_KnnBasictrain, elapsedtime_KnnBasictest, data):
    param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                    'min_support': [1, 5], 'user_based': [False]}}
    grid_search = GridSearch(KNNBasic, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    k = param['k']
    sim = param['sim_options']['name']
    min_support = param['sim_options']['min_support']
    user_based = param['sim_options']['user_based']
    for i in range(len(data)):
            training_start = time.time()
            training = data[i].build_full_trainset()
            testing = training.build_anti_testset()
            knn = KNNBasic(k=k, name=sim, min_support=min_support, user_based=user_based)
            knn.train(training)
            elapsedtime_KnnBasictrain.append(time.time() - training_start)

            test_start = time.time()
            knn.test(testing)
            elapsedtime_KnnBasictest.append(time.time() - test_start)


def knnm_running_time(elapsedtime_KnnMeanstrain, elapsedtime_KnnMeanstest, data):
    param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                    'min_support': [1, 5], 'user_based': [False]}}
    grid_search = GridSearch(KNNWithMeans, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    k = param['k']
    sim = param['sim_options']['name']
    min_support = param['sim_options']['min_support']
    user_based = param['sim_options']['user_based']

    for i in range(len(data)):
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        knnm = KNNWithMeans(k=k, name=sim, min_support=min_support, user_based=user_based)
        knnm.train(training)
        elapsedtime_KnnMeanstrain.append(time.time() - training_start)

        test_start = time.time()
        knnm.test(testing)
        elapsedtime_KnnMeanstest.append(time.time() - test_start)


def knnz_running_time(elapsedtime_KnnZtrain, elapsedtime_KnnZtest, data):
    param_grid = {'k': [5, 10, 20], 'sim_options': {'name': ['msd', 'cosine', 'pearson'],
                                                    'min_support': [1, 5], 'user_based': [False]}}
    grid_search = GridSearch(KNNWithZScore, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data[3])
    param = grid_search.best_params['RMSE']
    k = param['k']
    sim = param['sim_options']['name']
    min_support = param['sim_options']['min_support']
    user_based = param['sim_options']['user_based']

    for i in range(len(data)):
        training_start = time.time()
        training = data[i].build_full_trainset()
        testing = training.build_anti_testset()
        knnz = KNNWithZScore(k=k, name=sim, min_support=min_support, user_based=user_based)
        knnz.train(training)
        elapsedtime_KnnZtrain.append(time.time() - training_start)

        test_start = time.time()
        knnz.test(testing)
        elapsedtime_KnnZtest.append(time.time() - test_start)


if __name__ == "__main__":
    data = load_data()
    elapsedtime_SVDtrain = []
    elapsedtime_SVDtest = []
    svd_running_time(elapsedtime_SVDtrain, elapsedtime_SVDtest, data)

    elapsedtime_SVDpptrain = []
    elapsedtime_SVDpptest = []
    svdpp_running_time(elapsedtime_SVDpptrain, elapsedtime_SVDpptest, data)

    elapsedtime_NMFtrain = []
    elapsedtime_NMFtest = []
    nmf_running_time(elapsedtime_NMFtrain, elapsedtime_NMFtest, data)

    elapsedtime_Basetrain = []
    elapsedtime_Basetest = []
    base_running_time(elapsedtime_Basetrain, elapsedtime_Basetest, data)

    elapsedtime_KnnBasictrain = []
    elapsedtime_KnnBasictest = []
    knn_running_time(elapsedtime_KnnBasictrain, elapsedtime_KnnBasictest, data)

    elapsedtime_KnnMeanstrain = []
    elapsedtime_KnnMeanstest = []
    knnm_running_time(elapsedtime_KnnMeanstrain, elapsedtime_KnnMeanstest, data)

    elapsedtime_KnnZtrain = []
    elapsedtime_KnnZtest = []
    knnz_running_time(elapsedtime_KnnZtrain, elapsedtime_KnnZtest, data)

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