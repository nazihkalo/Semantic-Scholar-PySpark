
# # Graph Building & Analysis

import pandas as pd 
# import json
# import matplotlib.pyplot as plt 
from datetime import datetime
#Graph network imports
from graphframes import *
from pyspark import *
from pyspark.sql import *
import numpy as np
from pyspark.ml.linalg import *
from pyspark.ml.linalg import *
from pyspark.sql.types import * 
from pyspark.sql.functions import *
from functools import reduce
from pyspark.sql.functions import col, lit, when

from pyspark.sql.functions import udf #user defined function
from pyspark.sql.types import * #Import types == IntegerType, StringType etc.

#import statements
from pyspark.sql import SparkSession
#create Spark session
spark = SparkSession.builder.appName('SparkBasics').config("spark.jars.packages", "graphframes:graphframes:0.6.0-spark2.3-s_2.11").getOrCreate()
spark_df = spark.read.parquet('gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/encodedData/')

sc = spark.sparkContext

spark_df.printSchema()
spark_df.select('id', 'inCitations', 'outCitations').show()

# ### Creating Vertices

vertices = spark_df#.select('id', 'title', 'year', 'fieldsOfStudy', 'paperAbstract')
# ### Creating Edges

from pyspark.sql.functions import explode
from pyspark.sql.functions import lit

#Getting the inCitations id -> (cited) -> article id 
edges = spark_df.select(explode('inCitations').alias('src'), spark_df.id.alias('dst')).withColumn('type', lit('cited'))

#Getting the article id -> (cited) -> outCitations id
edges2 = spark_df.select(spark_df.id.alias('src'), explode('outCitations').alias('dst')).withColumn('type', lit('cited'))

#Union of these two 
edges_total = edges.union(edges2)

edges_total.show(5, truncate=False)



g = GraphFrame(vertices, edges_total)
## Take a look at the DataFrames
# g.vertices.show()
# g.edges.show()
## Check the number of edges of each vertex
# g.degrees.sort("degree", ascending=False).show()

# # #### indegree = The number of edges directed into a vertex in a directed graph.

# g.inDegrees.filter("inDegree >= 10").sort("inDegree", ascending=False).show()


# top_indegree = g.inDegrees.filter("inDegree >= 10").sort("inDegree", ascending=False).select('id').take(1)[0][0]


# top_node = g.vertices.filter('id == "{}"'.format(top_indegree)).toPandas()
# top_node


# #### Creating checkpoint directory in HDFS


sc.setCheckpointDir('/tmp/graphframes_cps')


# ## TRIANGLE COUNT

#Computes the number of triangles passing through each vertex.
# triangles = g.triangleCount()
# triangles.select('id', 'count')
# triangles.write.format('parquet').save('triangles')

# ## Looking at PageRank scores of nodes
# # Run PageRank
# print("STARTING PAGERANK at {}".format(datetime.now().strftime('%Y-%m-%d.%H:%M:%S')))
# results = g.pageRank(resetProbability=0.15, maxIter=10)
# pager = results.vertices.select("id", "pagerank")
# pager.write.format('parquet').save('gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/pagerank')

# pager_edges = results.edges.select('src', 'dst', 'weight')
# pager_edges.write.format('parquet').save('gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/pagerank_edges')
# ## look at the pagerank score for every vertex
# results.vertices.select("id", "pagerank").orderBy("pagerank", ascending=False).show(10)

# ## look at the pagerank score for every vertex
# results.vertices.show()
# ## look at the weight of every edge
# results.edges.show()

print("STARTING LabelPropogration at {}".format(datetime.now().strftime('%Y-%m-%d.%H:%M:%S')))
# ### Communities 
communities = g.labelPropagation(maxIter=5)
communities_save = communities.select('id', 'label')
communities_save.write.format('parquet').save('gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/communities')
communities.show()

print('END')