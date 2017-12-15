# personalization-theory-part2
 Group Members : **Archit Jain, Simran Lamba, Togzhan Sultan, TaeYoung Choi**

## Dataset

http://jmcauley.ucsd.edu/data/amazon/

-  Amazon product data: Video Games that contains 1,324,753 reviews
for 50,953 products.
-  Amazon product data: Electronics that contains 7,824,482 reviews for
498,196 products.

We explored the distribution of users, who submitted reviews after January 1st, 2011. As seen in the histogram below, we have an extremely skewed dataset with 75% of users submitted only single review and 95% of users submitted less than two reviews. Furthermore, 99% of users submitted less than four reviews. If we were to sample 10,000 users using this highly sparse
 data, we weren't able to meet the requirement, which is the size of the dataset being less than 100 items.

<table class="image">
<caption align="bottom">Histrogram of the number of reviews submitted by users</caption>
<tr><td><img src="https://github.com/taeyoung-choi/personalization-theory/blob/master/plot/byRatingNum.png" width="720"></td></tr>
</table>
