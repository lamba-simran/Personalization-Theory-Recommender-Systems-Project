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

Accuracy and coverage in tandem hold the most business value for a company like Amazon. For instance, the least popular items tend to provide the most profit if bought, due to higher margins. However, they are usually not recommended over popular items, as most recommender systems still put large focus only on accuracy, targeting only low hanging fruits.

We want the best mix of accuracy, converge, and serendipity to not only provide the best recommendation, but from a business perspective, the most profitable recommendations. We had initially chosen Video Games as the category to focus on as Steam has been at the forefront of Video Games recommendation and sales, and we strongly feel that it is a major category that can improve profitability. 

Targeting the electronics/accessories market was the next logical step as recommendations can then tie in with a category that is popular and extremely relevant to the aforementioned Video Games Category.

## Objective 

Recommendation systems have been used in e-commerce websites like Amazon to enable filtering through large observation and information space in order to provide recommendations in the information space that user does not have any observation. 

Our key objectives for this project is to operate on Amazon data set for Electronics/Accessories and Video Games and provide accurate, novel, and unique recommendation to users. Another key objective of this exercise is to design recommender systems and go into detail by exploring hyper-parameters to measure the impact of popular items or users. Another major objective is to dabble with technologies such as Spark and Python, further exploring various tools and libraries that are available and at our disposal. We believe that the amount of data that we are tackling (even after sampling) would require faster processing technology.

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

## Alternating Least Squares (ALS) Matrix Factorization method

ALS works by trying to find the optimal representation of a user and a product matrix – which when combined, should accurately represent the original dataset. The genius part of ALS is that it alternates between finding the optimal values for the user matrix and the product matrix. 


The implementation of ALS in spark.mllib has the following parameters:
- numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
-	rank is the number of latent factors in the model.
-	iterations is the number of iterations of ALS to run. ALS typically converges to a reasonable solution in 20 iterations or less.
-	lambda specifies the regularization parameter in ALS.
-	implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
-	alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.

## Locality Sensitive Hashing (LSH)

Since we are utilizing extremely sparse dataset, we decided to use LSH to map nearby data points to the same code by using hash functions that collide for similar points. For our purposes, we want a function that maps nearby points to the same hash value. To create recommendations, we utilized minhash function, as descripted in class in the class.

The first step of the process is to create a signature matrix composed of hash values that has a set of signatures for all items. The hash function takes the form [(a.item + b)%p], where b and p are prime, and operate on the item row index. Also, ‘p’ is a prime number that is greater than the maximum possible value of the number of items, a is any odd number that can be chosen between 1 and p-1, and b is any number that can be chosen from 0 to p-1. 

Then we will take the minimum hash value for each item and function, only considering the items that have been rated by at least due to the nature of our dataset. We thus create the signature matrix, that only contains the minimum values of hashed indices.

We then band the signature matrix into different bands and rows. For our purposes, we utilized a combination of different values of band size (number of bands) and hash size (the number of hash functions within each band). This will dictate whether the or not the items are hashed to the same bucket or not.

After the above steps have been completed, we then altered the process to obtain approximate ratings for all items that are in the same bucket, based on the ratings of the other items in the same bucket. To get the approximate average rating of an item, we used other items in the same bucket. We did this for all the items in our sample.






