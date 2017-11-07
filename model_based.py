import os
from surprise import Reader, Dataset, GridSearch, SVD, SVDpp, NMF, accuracy, BaselineOnly, KNNWithZScore, KNNWithMeans, KNNBasic
import numpy as np
import matplotlib.pyplot as plt


file_path = os.path.expanduser('sampled_data.csv')
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5) # data can now be used normally
training = data.build_full_trainset()
testing = training.build_anti_testset()

param_grid = {'n_factors': [25, 50, 100, 250], 'n_epochs': [10, 20, 30, 40, 50]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=False)
grid_search.evaluate(data)
param = grid_search.best_params['RMSE']
print('SVD:', param)
svd = SVD(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
svd.train(training)
predictions = svd.test(testing)
svd_rmse = accuracy.rmse(predictions, verbose=True)


grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE'], verbose=False)
grid_search.evaluate(data)
param = grid_search.best_params['RMSE']
print('SVDpp:', param)
svdpp = SVDpp(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
svdpp.train(training)
predictions = svdpp.test(testing)
svdpp_rmse = accuracy.rmse(predictions, verbose=True)

nmf_param_grid = {'n_factors': [45, 50, 55, 60], 'n_epochs': [45, 50, 55]}
grid_search = GridSearch(NMF, nmf_param_grid, measures=['RMSE'], verbose=False)
grid_search.evaluate(data)
param = grid_search.best_params['RMSE']
print('NMF:', param)
nmf = NMF(n_factors=param['n_factors'], n_epochs=param['n_epochs'])
nmf.train(training)
predictions = nmf.test(testing)
nmf_rmse = accuracy.rmse(predictions, verbose=True)

baseline = BaselineOnly()
baseline.train(training)
predictions = baseline.test(testing)
baseline_rmse = accuracy.rmse(predictions, verbose=True)


knn_param_grid = {'k': [20,30, 40],
             'sim_options': {'name': ['msd', 'cosine'],
                              'min_support': [5,10],
                              'user_based': [False]}
              }
knnz_grid_search = GridSearch(KNNWithZScore, knn_param_grid, measures=['RMSE'], verbose=False)
knnz_grid_search.evaluate(data)
param = knnz_grid_search.best_params['RMSE']
print('KNNWithZScore:', param)
knnz = KNNWithZScore(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
knnz.train(training)
predictions = knnz.test(testing)
knnz_rmse = accuracy.rmse(predictions, verbose=True)

knn_grid_search = GridSearch(KNNBasic, knn_param_grid, measures=['RMSE'], verbose=False)
knn_grid_search.evaluate(data)
param = knn_grid_search.best_params['RMSE']
print('KNNBasic:', param)
knn = KNNBasic(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
knn.train(training)
predictions = knnz.test(testing)
knn_rmse = accuracy.rmse(predictions, verbose=True)

knnm_grid_search = GridSearch(KNNWithMeans, knn_param_grid, measures=['RMSE'], verbose=False)
knnm_grid_search.evaluate(data)
param = knnm_grid_search.best_params['RMSE']
print('KNNWithMeans:', param)
knnm = KNNWithMeans(k=param['k'], name=param['sim_options']['name'], min_support=param['sim_options']['min_support'], user_based=param['sim_options']['user_based'] )
knnm.train(training)
predictions = knnm.test(testing)
knnm_rmse = accuracy.rmse(predictions, verbose=True)


objects = ("SVD", "SVD++", "NMF", "Baseline")
y_pos = np.arange(len(objects))
performance = [svd_rmse, svdpp_rmse, nmf_rmse, baseline_rmse]
plt.bar(y_pos, performance, align='center', alpha=0.1)
plt.xticks(y_pos, objects)
plt.show()
