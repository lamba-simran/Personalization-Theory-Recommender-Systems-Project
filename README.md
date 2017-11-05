# personalization-theory
 Group Members : **Archit Jain, Simran Lamba, Togzhan Sultan, TaeYoung Choi**

## Installation

## Dataset
Amazon product data : Video Games that contains 1,324,753 reviews for 50,953 products.


The video game categories that we decided to pick are PC, Mac, More Systems, Nintendo, PlayStation, multi support, Sony,
 Wii, and Xbox. The games that support different types of platforms were categorized as multi support. For simplicity,
 we did not distinguish different models for each platform. For example, Xbox means that a video game runs
 on one of the Xbox models such as Xbox 360, Xbox 360 S, Xbox 360 E, and etc. We eliminated all items that are outside
 of these categories, because they account for only a faction of data, and are not supported by any of the current
 platforms.

We assumed the reviews submitted after January 1st, 2011 were effective. The latest Nintendo model in 2009. The latest
Xbox model was released in 2010. Wii Family Edition was released in 2011, and the most recent version of PlayStation 3
was released in 2012, followed by PlayStation 4 in 2013. Any reviews submitted before these years seemed irrelevant, if
we were to give recommendations to current users.

We explored the distribution of users, who submitted reviews after January 1st, 2011. We have a extremely skewed dataset
 with 75% of users submitted only single review and 95% of users submitted less than two reviews. Furthermore, 99% of
 users submitted less than four reviews. If we were to sample 10,000 users using this highly sparse
 data, we weren't able to meet the requirement, which is the size of the dataset being less than 100 items.


However, we also do not treat heavy reviewers as same as inactive users, because these few heavy users creates bigger
impacts than normal users do. As a result, we adjusted our sampling method to sample users with single rating to account
 for approximately 25% of the data. By artificially feeding more data, we could better evaluate our model performances.
 Otherwise, all models perform poorly by having a n x n dataset with nearly n non-zero entries.
 
<table class="image">
<caption align="bottom">Histrogram of the number of reviews submitted by users</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/user_distribution.png" width="720"></td></tr>
</table>
 <table class="image">
<caption align="bottom">Histrogram of the number of reviews submitted by users</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/category_distribution.png" width="720"></td></tr>
</table>

## Objective

## Algorithms


## Misc
 Top n predictions code: https://pypkg.com/pypi/scikit-surprise/f/examples/top_n_recommendations.py
