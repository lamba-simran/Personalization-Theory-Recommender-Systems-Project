"""
Created on Oct 24, 2017
@author: TYchoi

"""

import random
import pandas as pd
import numpy as np


class MinHash(object):
    """
    Locality Sensitive Hashing using Minhash and recommends items using approximate nearest neighbors.

    Attributes:
        input               input matrix for training
        output              output matrix for validating
        hash_size           number of hash functions within each band
        band_size           number of bands
        row_num             number of rows of the input matrix
        col_num             number of columns of the input matrix
        hash_function       list of random hash functions of size hash_size * band_size
        signature_matrix    matrix that contains the minimum values of hashed indicies that user has consumed
        neighbors           list of users that shares the same hash value
        items               dictionary of items {number: item}. This serves as keys that we want to hash
        headers             list of category names
        result              dataframe that stores neighbors and recommended items for each user
    """
    def __init__(self, input_matrix, output_matrix, hash_size, band_size):
        self.input = input_matrix
        self.output = output_matrix
        self.hash_size = hash_size
        self.band_size = band_size
        self.hash_function = []
        self.signature_matrix = pd.DataFrame(columns=list(range(band_size)))
        self.neighbors = []
        self.user_num = self.input.shape[0]
        self.items = {}
        self.headers = list(input_matrix)
        self.result = pd.DataFrame(columns=['neighbors', 'item'])

        # unique items are keys we want to hash
        counter = 0
        for i in range(self.input.shape[0]):
            for k in self.input.iloc[i]['item']:
                if k not in self.items:
                    self.items[k] = counter
                    counter += 1

        self.item_num = len(self.items)

    def train(self):
        self.create_hash_functions()
        self.to_signature_matrix()
        self.neighboring()

    def predict(self, data):
        recom_list = []
        for i in range(data.shape[0]):
            user = data.iloc[i, :]
            neighbors, classes = self.find_neighbors(user)
            courses = self.recommend_items(classes)
            self.result.loc[i] = [neighbors, courses]
            recom_list += courses
        return set(recom_list)

    def accuracy(self, y, d, p):
        """
        calculates RMSE

        RMSE = sqrt(1/n * (predicted ratings - actual ratings)^2)
        """
        for i in range(y.shape[0]):
            for j in y.iloc[i]['rating']:
                if j in self.result.iloc[i]['item']:
                    d.append(self.result.iloc[i]['item'][j])
                    p.append(y.iloc[i]['rating'][j])

        rmse = np.sqrt(((np.array(d) - np.array(p))**2).mean())
        return rmse

    def create_hash_functions(self):
        function_list = []
        # p is a prime number that is greater than max possible value of x
        found = False
        p = self.item_num
        while not found:
            prime = True
            for i in range(2, p):
                if p % i == 0:
                    prime = False
            if prime:
                found = True
            else:
                p += 1
        # Corman et al as very readable information in section 11.3.3 pp 265-268.
        # https://mitpress.mit.edu/books/introduction-algorithms
        for i in range(self.band_size):
            for j in range(self.hash_size):
                # a is any odd number you can choose between 1 to p-1 inclusive.
                a = random.randrange(1, p, 2)
                # b is any number you can choose between 0 to p-1 inclusive.
                b = random.randint(0, p)
                function_list.append([a, b, p])
        self.hash_function = function_list

    def to_signature_matrix(self):
        """
        Creates Signature Matrix
        """
        sig_mat = []
        for i in range(self.user_num):
            user = self.input.iloc[i, :]
            each = []
            for func in self.hash_function:
                min_finder = []
                for category in user['item']:
                    key = self.items[category]
                    '''apply hash functions for the items that each user has rated'''
                    min_finder.append((func[0]*key + func[1]) % func[2])
                '''store min-hash'''
                each.append(min(min_finder))
            sig_mat.append(each)
        sig_mat = pd.DataFrame(sig_mat)

        for k in range(self.band_size):
            '''appropriately slicing min-hash values according to the band size and hash size'''
            start_index = k * self.hash_size
            end_index = start_index + self.hash_size
            subset_df = sig_mat.iloc[:, start_index:end_index]
            self.signature_matrix.iloc[:, k] = subset_df.to_records(index=False)

    def neighboring(self):
        """
        groups users according to signature matrix.
        If two users share the same values, they are put into the same group.
        """
        for k in range(self.band_size):
            hashed_items = {}
            for i in range(self.signature_matrix.shape[0]):
                key = self.signature_matrix.iloc[i, k]
                if key in hashed_items:
                    '''additional user with the same key'''
                    hashed_items[key].append(i)
                else:
                    '''new user with a unique key'''
                    hashed_items[key] = [i]
            self.neighbors.append(hashed_items)

    def find_neighbors(self, input_array):
        """
        input_array stores features for a user that we want to find neighbors of.
        applied the hash functions, we defined and compute min-hash values
        """
        specific_user = []
        for func in self.hash_function:
            min_finder = []
            for category in input_array['item']:
                if category not in self.items:
                    '''we might encounter new items that we have not seen in training dataset,
                       then assign unique item number'''
                    self.items[category] = self.item_num
                    self.item_num += 1

                key = self.items[category]
                min_finder.append((func[0] * key + func[1]) % func[2])
            specific_user.append(min(min_finder))

        neighbors = []
        for k in range(self.band_size):
            '''appropriately slicing min-hash values according to the band size and hash size'''
            start_index = k * self.hash_size
            end_index = start_index + self.hash_size
            each_band = tuple(specific_user[start_index:end_index])
            if each_band in self.neighbors[k]:
                '''go into the bin that user has been assign to and get neighbors'''
                neighbors = list(set(neighbors + self.neighbors[k][each_band]))

        var = list(self.output)[0]
        '''this stores items and ratings of neighbors'''
        neighbor_output = self.output.iloc[neighbors, :][[var]]

        average_ratings = {}
        for i in range(neighbor_output.shape[0]):
            '''getting average ratings of items that neighbors have given ratings to'''
            user_ratings = neighbor_output.iloc[i][[var]][0]
            for item in user_ratings:
                if item not in average_ratings:
                    average_ratings[item] = [user_ratings[item]]
                else:
                    average_ratings[item] = average_ratings[item] + [user_ratings[item]]
        for item in average_ratings:
            average_ratings[item] = sum(average_ratings[item])/float(len(average_ratings[item]))

        return neighbors, average_ratings


    def recommend_items(self, items):
        """
        giving top 5 courses with highest ratings from neighbors
        """
        s = [k for k in sorted(items, key=items.get, reverse=True)]
        # # top 1
        # s = s[:1]
        # # top 3
        # s = s[:3]
        # top 5
        s = s[:5]
        recommendation = {}
        for i in s:
            recommendation[i] = items[i]
        return recommendation
