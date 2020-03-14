#!/usr/bin/env python
# coding: utf-8

# ## Reading Data from HDFS
# 
# Datasource: Semantic Scholar Open Research Corpus
# 
# Description: Semantic Scholar's records for research papers published in all fields provided as an easy-to-use JSON archive.
# 
# ### Attribute Definitions
# 1. id  string = S2 generated research paper ID.
# - title  string = Research paper title.
# - paperAbstract  string = Extracted abstract of the paper.
# - entities  list = Extracted list of relevant entities or topics.
# - s2Url  string = URL to S2 research paper details page.
# - s2PdfUrl  string = URL to PDF on S2 if available.
# - pdfUrls  list = URLs related to this PDF scraped from the web.
# - authors  list = List of authors with an S2 generated author ID and name.
# - inCitations  list = List of S2 paper IDs which cited this paper.
# - outCitations  list = List of S2 paper IDs which this paper cited
# - year  int = Year this paper was published as integer.
# - venue  string = Extracted publication venue for this paper.
# - journalName  string = Name of the journal that published this paper.
# - journalVolume  string = The volume of the journal where this paper was published.
# - journalPages  string = The pages of the journal where this paper was published.
# - sources  list = Identifies papers sourced from DBLP or Medline.
# - doi  string = Digital Object Identifier registered at doi.org.
# - doiUrl  string = DOI link for registered objects.
# - pmid  string = Unique identifier used by PubMed.
# - fieldsOfStudy  list = Zero or more fields of study this paper addresses.`

# In[ ]:


import pandas as pd 
import json
#import matplotlib.pyplot as plt 

#Graph network imports
#from graphframes import *
from pyspark import *
from pyspark.sql import *
import numpy as np
from pyspark.ml.linalg import *
from pyspark.ml.linalg import *
from pyspark.sql.types import * 
from pyspark.sql.functions import *
import pyspark
from pyspark.sql.functions import udf #user defined function
from pyspark.sql.types import * #Import types == IntegerType, StringType etc.

import nltk


# In[ ]:


# spark-nlp components. Each one is incorporated into our pipeline.
from sparknlp.annotator import Lemmatizer, Stemmer, Tokenizer, Normalizer
from sparknlp.base import DocumentAssembler, Finisher


# In[ ]:


#import statements
from pyspark.sql import SparkSession
#create Spark session
spark = SparkSession.builder.appName('SparkBasics').config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.2.1").getOrCreate()


# ## Reading all 10 json files listed above 

# In[6]:


spark_df = spark.read.json('gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/unzipped_files/')
#spark.read.json('../../../scratch-midway2/big_data_project/unzipped_files/')


# ### Checking if any NaN rows 

# #### Counting number of records/articles

# ### Looks like all the NAs are in year, let's impute using the median year

# In[11]:


#Impute median year
year_stats = spark_df.select('year').summary()
median_year = int(year_stats.collect()[5].year)
print('the median year is = ', median_year)

#Inputing the median year
spark_df = spark_df.na.fill({'year': median_year})

#Convert to integer type column
spark_df = spark_df.withColumn("year", spark_df["year"].cast(IntegerType()))


# ## DROP NO TITLE & NO ABSTRACT

# #### Dropping those with blank abstract or title

# In[12]:


spark_df2 = spark_df.filter((col('title') != '') & (col('paperAbstract') != ''))

# ## Creating column that counts the inCitations = how many times this paper was cited

# In[13]:


from pyspark.sql.functions import udf #user defined function
from pyspark.sql.types import * #Import types == IntegerType, StringType etc.

length = udf(lambda listt: len(listt), IntegerType())


# In[14]:


spark_df2 = spark_df2.withColumn('inCitations_count', length(spark_df['inCitations']))
spark_df2.select('inCitations_count', 'inCitations').show(10)


# ## Creating column that counts the outCitations = how many other papers this paper cited

# In[15]:


spark_df2 = spark_df2.withColumn('outCitations_count', length(spark_df2['outCitations']))
spark_df2.select('outCitations_count', 'outCitations').show(10)


# ## Tokenize the abstract & title

# In[16]:


from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from nltk.stem.snowball import SnowballStemmer

# Clean text
spark_df2 = spark_df2.withColumn('paperAbstract', (lower(regexp_replace('paperAbstract', "[^a-zA-Z\\s]", ""))))
spark_df2 = spark_df2.withColumn('title', (lower(regexp_replace('title', "[^a-zA-Z\\s]", ""))))


# In[17]:


# word_count = udf(lambda string: len(nltk.word_tokenize(string)), IntegerType()) ## OLD SOLUTION 
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

#tokenize words
tokenizer = Tokenizer(inputCol="paperAbstract", outputCol="abstract_tokens")
spark_df2 = tokenizer.transform(spark_df2)

tokenizer = Tokenizer(inputCol="title", outputCol="title_tokens")
spark_df2 = tokenizer.transform(spark_df2)


# ## Remove Stopwords

# In[18]:


from pyspark.ml.feature import StopWordsRemover

#remove stop words
remover = StopWordsRemover(inputCol="abstract_tokens", outputCol="abstract_tokens_filtered")
spark_df2 = remover.transform(spark_df2)

#remove stop words
remover = StopWordsRemover(inputCol="title_tokens", outputCol="title_tokens_filtered")
spark_df2 = remover.transform(spark_df2)


# ## Filter short words

# In[19]:


# Filter length word > 3
filter_length_udf = udf(lambda row: [x for x in row if len(x) >= 3], ArrayType(StringType()))

spark_df2 = spark_df2.withColumn('abstract_tokens_filtered', filter_length_udf(col('abstract_tokens_filtered')))
spark_df2 = spark_df2.withColumn('title_tokens_filtered', filter_length_udf(col('title_tokens_filtered')))


# ## Lemmetize words

# In[20]:


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')
def stemFunct(x):
    finalStem = [stemmer.stem(s) for s in x]
    return finalStem

stem_udf = udf(lambda tokens: stemFunct(tokens), ArrayType(StringType())) 

spark_df2 = spark_df2.withColumn('abstract_tokens_filtered_lem', stem_udf(spark_df2['abstract_tokens_filtered']))
spark_df2 = spark_df2.withColumn('title_tokens_filtered_lem', stem_udf(spark_df2['title_tokens_filtered']))


# ## Create wcount columns

# In[21]:


from pyspark.sql.functions import udf 

word_count = udf(lambda tokens: len(tokens), IntegerType()) 

spark_df2 = spark_df2.withColumn('abstract_wcount', word_count(spark_df2['abstract_tokens_filtered_lem']))
spark_df2 = spark_df2.withColumn('title_wcount', word_count(spark_df2['title_tokens_filtered_lem']))

spark_df2.select('title_tokens_filtered_lem', 'title_wcount', 'abstract_tokens_filtered_lem', 'abstract_wcount').show(10)


# ## Dropping more rows with no title/abstract that got through first filtering

# In[22]:


spark_df2 = spark_df2.filter((col('title_wcount') != 0) & (col('abstract_wcount') != 0))


# In[ ]:


spark_df2.filter((col('title_wcount') == 0) | (col('abstract_wcount') == 0)).count()


# ## TFIDF for abstract and title

# In[ ]:


#Maps a sequence of terms to their term frequencies using the hashing trick. 
hashingTF1 = HashingTF(inputCol="abstract_tokens_filtered_lem", outputCol="abstract_tokens_filtered_lem_count")
hashingTF2 = HashingTF(inputCol="title_tokens_filtered_lem", outputCol="title_tokens_filtered_lem_count")
featurizedData = hashingTF1.transform(spark_df2)
featurizedData = hashingTF2.transform(featurizedData)

#Getting IDF
idf = IDF(inputCol="abstract_tokens_filtered_lem_count", outputCol="abstract_tfidf")
idf2 = IDF(inputCol="title_tokens_filtered_lem_count", outputCol="title_tfidf")

idfModel = idf.fit(featurizedData)
spark_df3 = idfModel.transform(featurizedData)
idfModel = idf2.fit(spark_df3)
spark_df3 = idfModel.transform(spark_df3)


# ## Word2Vec creating/loading Pyspark model

# In[ ]:


from pyspark.ml.feature import Word2Vec, Word2VecModel

#w2VM.save('word2vec_model_v50_min500_ws5_2.0')
w2VM = Word2VecModel.load("gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/word2vec_model_v50_min500_ws5_2.0")
nlpdf = w2VM.transform(spark_df3)


# ## Mapping SJR to journal name 

# In[ ]:


journal_map = spark.read.csv('gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/scimagojr_2017.csv', sep = ';',inferSchema=True, header=True )


# In[ ]:


#Renaming column so they dont have same name 
journal_map2 = journal_map.withColumnRenamed('Title', 'journal_name')


# In[ ]:


#Reduce to only relevant columns
journal_map2 = journal_map2.select('journal_name', 'SJR')


# In[ ]:


joined_dfs = nlpdf.join(journal_map2, on = nlpdf.journalName == journal_map2.journal_name, how = 'left_outer')


# In[ ]:


joined_dfs = joined_dfs.withColumn('SJR', (regexp_replace('SJR', ",", ".")).cast(FloatType()))


# ### Find 25 percentile to fill 

# In[ ]:


print('Before imputation : \n')
joined_dfs.select('SJR').summary().show()

joined_dfs = joined_dfs.na.fill({'SJR': 0.474})

print('After imputation : \n')
joined_dfs.select('SJR').summary().show()


# ## Fixing some dtypes
# 

# ### Splitting authors column into authorname & authorID

# In[ ]:


import pyspark.sql.functions as F
joined_dfs = joined_dfs.withColumn('author_ids', F.col('authors.ids'))
joined_dfs = joined_dfs.withColumn('author_names', F.col('authors.name'))


# ### Counting number of authors 

# In[ ]:


joined_dfs = joined_dfs.withColumn('author_count', word_count(joined_dfs['author_names']))

joined_dfs.select('author_names' ,'author_count').show(5)


# ## ONE HOT ENCODING CATEGORICALS

# ### Now let's unpack the arrays and turn them into strings

# In[ ]:


#Defining function to unpack list intro string joined with ', ' 
udf_unpack = udf(lambda listt: ', '.join(listt), StringType())


# In[ ]:


joined_dfs = joined_dfs.withColumn('fieldsOfStudy', udf_unpack(col('fieldsOfStudy')))
joined_dfs = joined_dfs.withColumn('sources', udf_unpack(col('sources')))


# In[ ]:


joined_dfs = joined_dfs.withColumn('sources', regexp_replace(col("sources"), r"^\s*$", "no_source"))
joined_dfs = joined_dfs.withColumn('fieldsOfStudy', regexp_replace(col("fieldsOfStudy"), r"^\s*$", "no_FoS"))


# In[53]:


joined_dfs.select('sources').distinct().show()


# ## Using StringIndexer -> OneHotEncoderEstimator to encode fieldsOfStudy & sources 

# In[ ]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator

#convert relevant categorical into one hot encoded
indexer1 = StringIndexer(inputCol="fieldsOfStudy", outputCol="fieldsOfStudyIdx").setHandleInvalid("skip")
indexer2 = StringIndexer(inputCol="sources", outputCol="sourcesIdx").setHandleInvalid("skip")

#gather all indexers as inputs to the One Hot Encoder
inputs = [indexer1.getOutputCol(), indexer2.getOutputCol()]

#create the one hot encoder
encoder = OneHotEncoderEstimator(inputCols=inputs,                                   outputCols=["fieldsOfStudyVec", "sourcesVec"])

#run it through a pipeline
pipeline = Pipeline(stages=[indexer1, indexer2, encoder])
encodedData = pipeline.fit(joined_dfs).transform(joined_dfs)

#we have removed NAs so dont need to impute missing values.
#pipeline = pipeline.na.fill(0) 

encodedData.select('fieldsOfStudy', 'fieldsOfStudyVec', 'sources', 'sourcesVec').show(5)


# ## Saving the relevant columns of the feature engineered dataframe

# In[ ]:


drop_cols = ['authors','doi','doiUrl','entities','journalPages',
 'journalVolume',
 'pdfUrls',
 'pmid',
 's2PdfUrl',
 's2Url','abstract_tokens',
 'title_tokens',
 'abstract_tokens_filtered',
 'title_tokens_filtered',
 'abstract_tokens_filtered_lem',
 'title_tokens_filtered_lem',
 'abstract_tokens_filtered_lem_count',
 'title_tokens_filtered_lem_count']


# In[ ]:


encodedData_save = encodedData.drop(*drop_cols)


# In[ ]:


encodedData_save.write.format('parquet').save('gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/encodedData')


# ## SELECTING COLUMNS FOR MODELING

# In[89]:


modeling_data = encodedData.select(
    'id', 'year',
 'inCitations_count',
 'outCitations_count','abstract_wcount',
 'title_wcount','abstract_tfidf',
 'title_tfidf',
 'SJR','author_count','fieldsOfStudyVec',
 'sourcesVec', 'title_wordVectors')


# In[90]:


from pyspark.ml.feature import VectorAssembler
#gather feature vector and identify features
assembler = VectorAssembler(inputCols = ['year', 'outCitations_count','abstract_wcount','title_wcount','abstract_tfidf',                                         'title_tfidf','SJR','author_count','fieldsOfStudyVec', 'sourcesVec',                                          'title_wordVectors'],                             outputCol = 'features')


modeling_data = assembler.transform(modeling_data)


# In[ ]:


modeling_data.write.format('parquet').save('gs://dataproc-0f46e279-d9a7-4ef1-b8ee-3ba36c28428d-us-central1/modeling_data2')


# In[ ]:


print('FINISHED')

