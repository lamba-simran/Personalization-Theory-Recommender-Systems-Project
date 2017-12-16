"""
Created on Nov 14, 2017
@author: TYchoi

"""
import math
import numpy as np
import LSH
import pandas as pd


class CrossValidate(object):
    """
    Evaluating Algorithms using n-folds cross validation

    Attributes:
        data_folds          store data splitted into n-folds
        n_folds             number of folds
        input               input data
        output              observed output for the given input data
        row_num             number of rows of the input matrix
        recommended         list of recommended items
    """
    def __init__(self, input, output, n_folds=5):
        self.data_folds = {}
        self.n_folds = n_folds
        self.input = input
        self.output = output
        self.recommended = []

    def split(self):
        """
        split data into n folds with equal sizes
        """
        nrows = self.input.shape[0]
        numbers = list(range(nrows))
        each_fold_size = math.floor(float(nrows)/self.n_folds)
        for i in range(1, self.n_folds):
            if i != self.n_folds:
                self.data_folds[i] = np.random.choice(numbers, each_fold_size, replace=False).tolist()
                for item in self.data_folds[i]:
                    numbers.remove(item)
        self.data_folds[self.n_folds] = numbers

    def evaluate_minhash(self, b, r):
        """
        evaluates Min-hash
        """
        mean_accuracy = 0
        mean_coverage = 0
        headers = []
        for i in range(1, self.n_folds + 1):
            headers.append("fold {}".format(i))
        headers.append("Mean")

        accuracy_list = []
        coverage_list = []
        for i in range(1, self.n_folds+1):
            print("fold {}".format(i))
            d = []
            p = []
            indices = self.data_folds[i]
            training = self.input.drop(self.input.index[indices])
            training_y = self.output.drop(self.output.index[indices])
            test = self.input.loc[self.input.index[indices], :]
            test_y = self.output.loc[self.output.index[indices], :]
            '''train without fold i'''
            lsh = LSH.MinHash(training, training_y, b, r)
            lsh.train()
            '''test on fold i'''
            courses = lsh.predict(test)
            '''calculate rmse'''
            rmse = lsh.accuracy(test_y, d, p)
            accuracy_list.append(rmse)
            mean_accuracy += rmse
            '''calculate coverage. We have defined coverage as follows:
               coverage = # of unique items we have recommended on the test set / # of all items
            '''
            for item in courses:
                if item not in self.recommended:
                    self.recommended.append(item)
            c = len(self.recommended)/float(lsh.item_num)
            mean_coverage += c
            coverage_list.append(c)
        coverage_list.append(float(mean_coverage)/self.n_folds)
        accuracy_list.append(float(mean_accuracy)/self.n_folds)
        accuracy_table = pd.DataFrame([accuracy_list, coverage_list], columns=headers)
        accuracy_table = accuracy_table.rename(index={0: "RMSE", 1:"Coverage"})
        print(accuracy_table)

        return accuracy_table
