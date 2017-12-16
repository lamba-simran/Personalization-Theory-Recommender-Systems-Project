
# In[ ]:

from pyspark.sql import SparkSession
from pymining import itemmining, assocrules, perftesting
import pyfpgrowth
spark = SparkSession.builder.master("local").appName("Linear Regression Model").config("spark.executor.memory", "1gb").getOrCreate()
print(type(spark))
csv = spark.read.csv("/Users/archit/AnacondaProjects/Neighbor/ratings_Electronics_1M.csv")
csv = csv.selectExpr("_c0 as user", "_c1 as item", "_c2 as rating", "_c3 as time")
distinct_val = csv.select('item').distinct()
# In[ ]:

from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, LongType

with_index = distinct_val.rdd.zipWithIndex()
distinct_val = with_index.map(lambda x: (Row(item = x[0].item, id =x[1])))
distinct_val = distinct_val.toDF()


# In[ ]:

joined_df = csv.join(distinct_val, ['item'], "left")
rdd1 = joined_df.rdd


# In[ ]:

rdd2 = rdd1.map(lambda line: (line.user, (line.id, line.rating))).groupByKey().cache()


# In[ ]:

def sv_format2(x):
    item = []
    rating = []
    for i in x:
        item.append(i[0])
    sorted_points = sorted(item)
    return sorted_points

sparseVectorData2 = rdd2.map(lambda a :sv_format2(a[1]))

transactions = sparseVectorData2.collect()

#print sparseVectorData

#Frequent Item Set Mining
relim_input = itemmining.get_relim_input(transactions)
report = itemmining.relim(relim_input, min_support=10)
print report
#Association Rules
rules = assocrules.mine_assoc_rules(report, min_support=10, min_confidence=0.1)
print rules

#==============================================================================
# patterns = pyfpgrowth.find_frequent_patterns(transactions, 10)
# print patterns
# 
# rules = pyfpgrowth.generate_association_rules(patterns, 0.5)
# print rules
#==============================================================================


