import pandas as pd
import Accuracy
import LSH
import pickle
import time
import pprint
import ast
import math
import numpy as np



def main():
    df = pd.read_csv("old.csv", names=['user', 'rating'])
    df2 = pd.read_csv("new.csv", names=['user', 'rating'])

    df['rating'] = df.apply(lambda row: ast.literal_eval(row['rating']), axis=1)
    df2['rating'] = df2.apply(lambda row: ast.literal_eval(row['rating']), axis=1)
    item_column = []
    item_column2 = []
    for i in range(df.shape[0]):
        item_column.append(list(df.iloc[i]['rating']))

    for i in range(df2.shape[0]):
        item_column2.append(list(df2.iloc[i]['rating']))
    df['item'] = pd.Series(item_column)
    df2['item'] = pd.Series(item_column2)
    # df['item'] = df.apply(lambda row: list(row['rating'].keys()), axis=1)
    output = df[['rating']]
    input = df[['item']]
    input2 = df2[['item']]
    # print('data loaded')
    # data = Accuracy.CrossValidate(input, output, n_folds=5)
    # data.split()
    # print('data preprocessed')
    # tuned_param = list()
    # for i in range(4, 5):
    #     for j in range(3, 4):
    #         print(i, j)
    #         accuracy = data.evaluate_minhash(i, j)
    #         mean_score = accuracy['Mean'][0]
    #         if len(tuned_param) == 0:
    #             tuned_param = [i, j, mean_score]
    #         elif tuned_param[2] > mean_score:
    #             tuned_param = [i, j, mean_score]
    # print("best param: ", tuned_param[0], tuned_param[1])

    nrows = input.shape[0]
    numbers = list(range(nrows))
    each_fold_size = math.floor(float(nrows) / 5)
    indices = np.random.choice(numbers, each_fold_size, replace=False).tolist()

    nrows2 = input2.shape[0]
    numbers2 = list(range(nrows2))
    each_fold_size2 = math.floor(float(nrows2) / 5)
    indices2 = np.random.choice(numbers2, each_fold_size2, replace=False).tolist()

    training = input.drop(input.index[indices])
    training_y = output.drop(output.index[indices])
    test1 = input.loc[input.index[indices], :]
    test2 = input2.loc[input2.index[indices2], :]

    lsh = LSH.MinHash(training, training_y, 4, 3)
    lsh.train()
    '''test on fold i'''
    courses = lsh.predict(test1)
    courses2 = lsh.predict(test2)

    counter = 0
    for i in courses2:
        if i not in courses:
            counter += 1

    print (counter/float(len(courses2)))


if __name__ == "__main__":
    main()
