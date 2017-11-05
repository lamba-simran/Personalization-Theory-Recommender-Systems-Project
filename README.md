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

We explored the distribution of users, who submitted reviews after January 1st, 2011. As seen in the histogram below, we have a extremely skewed dataset with 75% of users submitted only single review and 95% of users submitted less than two reviews. Furthermore, 99% of users submitted less than four reviews. If we were to sample 10,000 users using this highly sparse
 data, we weren't able to meet the requirement, which is the size of the dataset being less than 100 items.

<table class="image">
<caption align="bottom">Histrogram of the number of reviews submitted by users</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/user_distribution.png" width="720"></td></tr>
</table>

However, we also did not want to treat heavy reviewers as same as inactive users, because these few heavy users creates bigger
impacts than normal users do. Therefore, we adjusted our sampling method to sample users with single rating to account
 for approximately 25% of the data. By artificially feeding more data, we could better evaluate our model performances.
 Otherwise, all models perform poorly by having a n x n dataset with nearly n non-zero entries. As a result, we sampled 20,000 users from the population. First 5,000 users were sampled from the group of users that submitted less than 2 reviews. Next 5,000 users were sampled from the group of users that submitted between 2 and 4 reviews. Another 5,000 users were sampled from the group that submitted between 4 and 7 reviews. The last 5,000 users were the sample from the group that submitted more than 7 reviews.
 
After sampling users, we sampled items according to the distribution of the video game categories that we defined earlier. The histrogram below represents the distribution of video game categories. For example, we randomly sampled 34 PC games, because the overall proportion of PC games in the population was 34%. In addition, we sampled according to the popularities of video games. For example, 70%, 85% and 90% of PC games recieved less than 3, 5 and 7 reviews, respectively. In order to overcome the sparcity issue within the items, we also adjusted the popularity measure. Finally, we sampled 9 random PC games with less than 3 reviews, 9 random PC games with between 3 and 5 reviews, 9 random games with between 5 and 7 reviews and 11 random games with more than 7 reviews. 

<table class="image">
<caption align="bottom">Histrogram of the number of reviews submitted by users</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/category_distribution.png" width="720"></td></tr>
</table>

This sampling method resulted in the sample of 100 items and 2,916 users by incorporating the distribution of the level of activities of users and the distribution of popularity and category of items.


## Objective

## Algorithms


## Misc
 Top n predictions code: https://pypkg.com/pypi/scikit-surprise/f/examples/top_n_recommendations.py
