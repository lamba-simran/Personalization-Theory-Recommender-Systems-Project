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

Our key objectives for this project is to operate on Amazon data set for Electronics/Accessories and Video Games and provide accurate, novel, and unique recommendation to users. Another key objective of this exercise is to design recommender systems and go into detail by exploring hyper-parameters to measure the impact of popular items or users. Another major objective is to dabble with tolls such as Spark and Python, further exploring various tools and libraries that are available and at our disposal.

For this purpose, we have used time-stamped user rating data from Amazon, which cumulatively have close to 10 million user-item ratings. The source for the data can be found at: http://jmcauley.ucsd.edu/data/amazon/links.html


## Dataset

For this purpose, we have used user rating data from Amazon, which cumulatively have close to 9 million user-item ratings. The source for the data can be found at: http://jmcauley.ucsd.edu/data/amazon/links.htmlhttp://jmcauley.ucsd.edu/data/amazon/

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


