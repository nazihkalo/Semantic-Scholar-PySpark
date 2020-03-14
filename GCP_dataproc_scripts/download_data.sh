

wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-02-01/manifest.txt
wget -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-02-01/ -i manifest.txt


wget -qO- -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-01-13/ -i manifest.txt | hdfs dfs -put - /user/$USER/data/manifest.txt 

gunzip | hdfs dfs -put

wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-02-01/manifest.txt
wget -qO- -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-01-13/ -i manifest.txt | hdfs dfs -put - /user/$USER/data
hdfs dfs -chmod 777 big_data_project/s2-corpus-00*.gz
hdfs dfs -cat big_data_project/s2-corpus-00[123456].gz | gzip -d | hdfs dfs -put - big_data_project/s2-corpus-001.txt

wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-02-01/manifest.txt
wget -qO- -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-01-13/ -i manifest.txt | hdfs dfs -put - /user/$USER/big_data_project
hdfs dfs -chmod 777 big_data_project/s2-corpus-*.gz
hdfs dfs -cat big_data_project/s2-corpus-*.gz | gunzip -d | hdfs dfs -put - big_data_project/


hdfs dfs -cat big_data_project/*.json | gsutil -m cp -r - gs://semantic_scholar_files/unzipped_files/