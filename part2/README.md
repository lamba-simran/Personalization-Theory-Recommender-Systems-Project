# personalization-theory-part2
 Group Members : **Archit Jain, Simran Lamba, Togzhan Sultan, TaeYoung Choi**


## Problem Background

In April 2017, Amazon stated that net sales rose 23 percent to $35.7 billion, beating analysts’ average estimate of $35.3 billion. Furthermore, Net income rose 41 percent over last year to $724 million. 

However, the key reason Amazon’s revenue has increased in recent years has been that businesses have moved their computing operations to the cloud, where Amazon Web Services (AWS) is the biggest player. Sales from AWS, the company’s fast-growing business to host companies’ data and handle their computing in the cloud, rose 42.7 percent to $3.66 billion. 

AWS accounts for a majority of Amazon’s operating profit. Seattle-based Amazon forecast that operating income in the second quarter would be between $425 million and $1.075 billion, below the average estimate of $1.46 billion. In fact, calculations show that Amazon’s retail e-commerce business has been loss making in majority of the past 12 quarters. 

In October 2016, Amazon posted $575 million in operating income for the third quarter in 2016, with help from $861 million in operating income from Amazon Web Services. The cloud-computing division’s $3.2 billion in revenue accounted for nearly 10 percent of Amazon’s companywide revenue of $32.7 billion. AWS’s operating income was 150 percent of Amazon’s operating income as a whole, helping offset a $541 million operating loss in the company’s international segment.

Even though Amazon is still currently focused on growth for both of its two main businesses, it is crucial for it to arrest its losses within the e-commerce business.


<table class="image">
<caption align="bottom">Amazon profit by category</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/Amazon_Profit.png" width="720"></td></tr>
</table>

## Problem Statement

Accuracy, coverage, novelty in tandem hold the most business value for a company like Amazon. For instance, the least popular items tend to provide the most profit if bought, due to higher margins. However, they are usually not recommended over popular items, as most recommender systems still put large focus only on accuracy, targeting only low hanging fruits.

We want the best mix of accuracy, converge, and novelty to not only provide the best recommendation, but from a business perspective, the most profitable recommendations. We had initially chosen Video Games as the category to focus on as Steam has been at the forefront of Video Games recommendation and sales, and we strongly feel that it is a major category that can improve profitability. 

Targeting the electronics/accessories market was the next logical step as recommendations can then tie in with a category that is popular and extremely relevant to the aforementioned Video Games Category.

## Objective 

Recommendation systems have been used in e-commerce websites like Amazon to enable filtering through large observation and information space in order to provide recommendations in the information space that user does not have any observation. 

Our key objectives for this project is to operate on Amazon data set for Electronics/Accessories and Video Games and provide accurate, novel, and varied recommendation to users. We want to ensure that we maintain the accuracy, while improving novelty if 

Another key objective of this exercise is to design recommender systems and go into detail by exploring hyper-parameters to measure the impact of popular items or users. Another major objective is to dabble with technologies such as Spark and Python, further exploring various tools and libraries that are available and at our disposal. We believe that the amount of data that we are tackling (even after sampling) would require faster processing technology.

For this purpose, we have used time-stamped user rating data from Amazon, which cumulatively have close to 10 million user-item ratings. The source for the data can be found at: http://jmcauley.ucsd.edu/data/amazon/links.html


## Dataset

For this purpose, we have used user rating data from Amazon, which cumulatively have close to 9 million user-item ratings. The source for the data can be found at: http://jmcauley.ucsd.edu/data/amazon/links.htmlhttp://jmcauley.ucsd.edu/data/amazon/

-  Amazon product data: Video Games that contains 1,324,753 reviews
for 50,953 products.
-  Amazon product data: Electronics that contains 7,824,482 reviews for
498,196 products.

We explored the distribution of users, who submitted reviews after January 1st, 2011. As seen in the histogram below, we have an extremely skewed dataset with 75% of users submitted only single review and 95% of users submitted less than two reviews. Furthermore, 99% of users submitted less than four reviews.

<table class="image">
<caption align="bottom">Histrogram of the number of reviews submitted by users</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/byRatingNum.png" width="720"></td></tr>
</table>

## Benchmark

Based on the for Part 1 of the project, we believed that that SVD++ is the best model to implement for the that sample because it has the best accuracy and does better than the others. Even though running time for training is relatively big and the testing time is about the average across all previous models, we think that training could be done offline and thus overall running will time not be affected. Furthermore, the Coverage for SVD++ was close to 30%, which we find sufficient for this data, while RMSE was close to 0.25.

Hence, we chose this model from all of our models as the baseline/benchmark to compare our new models to. Keeping in line with the objectives, coverage and novelty trump accuracy for this part of the project as we want to ensure that we are able to increase profits of the suppliers (and hence Amazon) by recommending more profitable, less popular items. We would like to improve the coverage and novelty as compared to baseline data, and are ready to accept a decrease in accuracy if needed, while keeping tabs on it all the same.

## Algorithms

We decided to build, explore and compare two recommendation models: Locality Sensitive Hashing (LSH), from scratch using python, and ALS Matrix Factorization in Pyspark.

## [Alternating Least Squares (ALS) Matrix Factorization method](https://github.com/taeyoung-choi/personalization-theory/blob/master/part2/ALS.ipynb)

ALS works by trying to find the optimal representation of a user and a product matrix – which when combined, should accurately represent the original dataset. The genius part of ALS is that it alternates between finding the optimal values for the user matrix and the product matrix. 


The implementation of ALS in spark.mllib has the following parameters:
- numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
-	rank is the number of latent factors in the model.
-	iterations is the number of iterations of ALS to run. ALS typically converges to a reasonable solution in 20 iterations or less.
-	lambda specifies the regularization parameter in ALS.
-	implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
-	alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.

Since ALS only accepts numeric datatypes, we had to change the userIDs and itemIDs from strings to numeric.

Changing the userID from string to numeric:

```python

ratings1=ratings.select('userID').distinct()
ratings1.show()

```
```
+--------------+
|        userID|
+--------------+
| AFS5ZGZ2M3ZGV|
|A18FTRFQQ141CP|
|A2GPNXFUUV51ZZ|
|A3GF7FD6065R2H|
|A16W4IDX9O70NU|
| A4035XND6J8CS|
|A15K7HV1XD6YWR|
|A3FZ6D8NP9775P|
| AA36DB7PNNJP2|
|A3PDGWYC08DXF4|
|A37JOONBIY5POU|
|A29XPB4YTMCH7N|
| AC8WXA642LUCJ|
|A2S7O09DKY30TH|
|A1G37NGYG23QG2|
|  A44UKZE6XEV9|
| AQJQP7RPKQGI9|
|A3BDHHIUI08PXH|
|A1LT2ILVBRDMN2|
|A3SAN0HBKL4O51|
+--------------+
only showing top 20 rows
```
```python
with_index = ratings1.rdd.zipWithIndex()
distinct_val = with_index.map(lambda x: (Row(userID = x[0][0], id =x[1])))
distinct_val = distinct_val.toDF()

#ratings_df = sqlContext.read.format('com.databricks.spark.csv').options(header=True, inferSchema=False).schema(ratings_df_schema).load(ratings_filename)
#ratings_df = raw_ratings_df.drop('Timestamp')
#ratings_df.cache()
distinct_val.show()
```
```
+---+--------------+
| id|        userID|
+---+--------------+
|  0| AFS5ZGZ2M3ZGV|
|  1|A18FTRFQQ141CP|
|  2|A2GPNXFUUV51ZZ|
|  3|A3GF7FD6065R2H|
|  4|A16W4IDX9O70NU|
|  5| A4035XND6J8CS|
|  6|A15K7HV1XD6YWR|
|  7|A3FZ6D8NP9775P|
|  8| AA36DB7PNNJP2|
|  9|A3PDGWYC08DXF4|
| 10|A37JOONBIY5POU|
| 11|A29XPB4YTMCH7N|
| 12| AC8WXA642LUCJ|
| 13|A2S7O09DKY30TH|
| 14|A1G37NGYG23QG2|
| 15|  A44UKZE6XEV9|
| 16| AQJQP7RPKQGI9|
| 17|A3BDHHIUI08PXH|
| 18|A1LT2ILVBRDMN2|
| 19|A3SAN0HBKL4O51|
+---+--------------+
only showing top 20 rows
```
We similarly change the itemID from string to numeric. 

Next, we transform the dataframe into a RDD and further spilt it into train set and test set. The model is fit on the train set and tuned using a number of parameter combinations. We found the best rmse for the following parameters:

For Product X, we find N Users to Sell To:

```python
model.recommendUsers(242,10)
```

For User Y, we find N Products to Promote:

```python
model.recommendProducts(196,10)
```
After making predictions for all ratings, we evaluated the model first on the train set and then on the test set. The results were as follows:

**Train Set RMSE: 0.5608492447656984**

**Test Set RMSE: 1.6515262287005992**

We also tried to evaluate **catalog coverage** for our model which came out be as low as **3.848 %** since coverage is impacted to a greater degree when non-popular items are not rated/recommended for any users and ALS recommends popular items over and over again. 

In order to get a better coverage, we decided to implement a type of **Approximate Nearest Neighbors** recommendation algorithm.


## [Locality Sensitive Hashing (LSH)](https://github.com/taeyoung-choi/personalization-theory/blob/master/part2/LSH.py)

Due to the size of the dataset, we decided to use pyspark and its parallelized computing features to process the data. We first converted the database into RDD objects and then grouped ratings by user, with each row being all the ratings for all the items that the user has rated. This gives us a very convenient data object to utilize for the purposes of LSH execution. Then we randomly shuffled users so that we can then splice the data into different samples and run a permutation of different parameters, using different sample sizes. 

Since we are utilizing extremely sparse dataset, we decided to use LSH to map nearby data points to the same code by using hash functions that collide for similar points. For our purposes, we want a function that maps nearby points to the same hash value. To create recommendations, we utilized minhash function, as descripted in class, and get the nearest neighbors.

The first step of the process is to create a signature matrix composed of hash values that has a set of signatures for all users. The hash function takes the form [(a(item) + b)%p], and operate on the item row index. Also, ‘p’ is a prime number that is greater than the maximum possible value of the number of items, a is any odd number that can be chosen between 1 and p-1, and b is any number that can be chosen from 0 to p-1. This method has been introduced by [Corman et al](https://mitpress.mit.edu/books/introduction-algorithms).

```python
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
a = random.randrange(1, p, 2)
b = random.randint(0, p)
```

We then band the signature matrix into different bands and rows. For our purposes, we utilized a combination of different values of band size (number of bands) and hash size (the number of hash functions within each band). This will dictate the probability that two items collide in the same bucket. We found the optimal values for band size and hash size based on the RMSE scores calculated using 5-fold cross validation.

```python
data = Accuracy.CrossValidate(input, output, n_folds=5)
data.split()
    for i in range(2, 5):
        for j in range(2, 5):
            print(i, j)
            accuracy = data.evaluate_minhash(i, j)
            mean_score = accuracy['Mean'][0]
            if len(tuned_param) == 0:
                tuned_param = [i, j, mean_score]
            elif tuned_param[2] > mean_score:
                tuned_param = [i, j, mean_score]
    print("best param: ", tuned_param[0], tuned_param[1])
```

After the above steps have been completed, we then altered the process to obtain approximate ratings for all items that are in the same bucket, based on the ratings of the other items in the same bucket. To get the approximate average rating of an item, we used other items in the same bucket. We did this for all the items in our sample. we then get the top-k (k=1,3,5) predictions for each user to calculate RMSE and coverage parameters. Once we had the average ratings of the top , we then compared them to the actual ratings of the items to calculate the RMSE. To calculate the coverage, we took the ratio of total number of unique items recommended upon the total number items in the sample utilized.

```python
for i in range(y.shape[0]):
    for j in y.iloc[i]['rating']:
        if j in self.result.iloc[i]['item']:
            d.append(self.result.iloc[i]['item'][j])
            p.append(y.iloc[i]['rating'][j])

rmse = np.sqrt(((np.array(d) - np.array(p))**2).mean())
```

Below are the results for the LSH execution for over 100,000 users. This was done to evaluate the performance of the model based on the Hash Size and the Band Size. We changes the values of those parameters and obtained the following results: 

<table class="image">
<caption align="bottom">LSH RMSE k=5</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/LSH_RMSE_Values.png" width="720"></td></tr>
</table>

|RMSE         |Band Size = 2|Band Size = 3  |Band Size = 4 |
|-------------|-------------|:-------------:|-------------:|
Hash Size = 2 |1.646	       |1.652	         |1.645         |
Hash Size = 3 |1.642	       |1.646	         |1.642         |
Hash Size = 4	|1.642        |1.641	         |1.641         |

<table class="image">
<caption align="bottom">LSH Coverage k=5</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/LSH_Coverage_Values.png" width="720"></td></tr>
</table>

|Coverage     |Band Size = 2|Band Size = 3  |Band Size = 4 |
|-------------|-------------|---------------|-------------:|
Hash Size = 2 |0.286	       |0.521	         |0.534         |
Hash Size = 3 |0.437	       |0.525	         |0.535         |
Hash Size = 4	|0.499        |0.532	         |0.536         |

Once the above is implemented, we observed that teh best combination (for accuracy and coverage) was for Hash Size and Band Size being 4 and 3 respectively. We used that to calculate the accuracy and coverage by increasing the sample to 1M ratings and observed that the RMSE remains relatively same at 1.63, but coverage reduces to **18.26%**. However, in absolute terms, this implies that over **200K** unique items are being recommended, as compared to only around 50% in the previous case.

One of the most encouraging results of the above was the fact that our novelty was **77.58%**. We had split our sample 80/20 intro train and test_old, then used a test_new (same size as test_old). Then we got the top k recommendations for both the test data sets. For k=5 on using above data, we found that the number 77.58% of the recommendations for the test_new set were different (and novel) than the recommendations in test_old. LSH ws extremely impressive in providing new recommendations to new users

## [Extension of the Model – Creating a hybrid model (LSH, FPM/Association Rules)](https://github.com/taeyoung-choi/personalization-theory/blob/master/part2/FPM.py)

We observed that even though our coverage has improved, there has been an overall decline in accuracy. To counter this, we tried to implement a hybrid approach of LSH and an undirected data mining technique in the form of Frequent Pattern Mining and Association Rules.

Since we could not implement Association Rules in pyspark, we decided to use pyspark only for data processing and then utilize python for the purpose of executing FPM and Association Rules. The key parameters to consider while creating item sets was the minimum support level (i.e. the minimum number of times a set of items appears in the code) as well as the confidence level for an association rule (the ratio of the number of transactions in which the item set appears).

```python

#Frequent Item Set Mining
relim_input = itemmining.get_relim_input(transactions)
report = itemmining.relim(relim_input, min_support=5)
print report

#Association Rules
rules = assocrules.mine_assoc_rules(report, min_support=5, min_confidence=0.1)
print rules

```

The main intuition behind this was to try and provide more recommended items in conjunction with LSH recommendations. For instance, let’s say that a particular user was recommended Item1, Item2 and Item3, (where Item3 is not a particularly accurate recommendation, and has a predicted average rating less than a certain threshold) by the LSH model, while the item set {Item1, Item2, Item42, Item52} has the desired minimum support and confidence levels, then we can replace Item3 with Item42 or Item 52; while increasing the accuracy and maintaining the coverage, while increasing novelty. We realized that we may not find too many such instances, but wanted to executre nonetheless.

The plan was to run all the recommendations for all the Users by the frequent item sets obtained from the FPM implementation with minimum support ranging close to 5 transactions and confidence close to 0.1. However, we were only able to implement FPM and unable to create a hybrid model. We believe that using a hybrid approach, we would have improved the total accuracy of the model. 


## References


Hash Function : https://mitpress.mit.edu/books/introduction-algorithms

FPM MLLib:http://spark.apache.org/docs/2.2.0/mllib-frequent-pattern-mining.html

FPM:https://fp-growth.readthedocs.io/en/latest/readme.html

ALS: http://www.learnbymarketing.com/644/recsys-pyspark-als/
