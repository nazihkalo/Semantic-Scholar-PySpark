# Semantic-Scholar-PySpark

Utilizing Spark and other Big Data technologies to analyze Semantic Scholar's corpus of over ~175M research papers. Built a knowledge graph using articles as nodes and citations as edges and attempted to discover what features are important in getting cited. 

## Dataset
Datasource https://api.semanticscholar.org/corpus/

## Graphs

We built a knowledge graph of the research papers utilizing Spark Graphframes, networkx, & pyvis for visualization. The initial graph had articles as nodes and citations as edges. In order to get a more manageable graph size and a visual understanding of the connections between papers we decided to group papers by their field of study and use the number of citations as the weights of the edges.

We then applied various clustering/community building algorithms on these graphs to create clusters and understand relationships between fields of study. Below are visual representations of these communities/clusters.

![Label Propogation Graph](images_gifs/LabelPropogation_Graph.gif)

![Strongly Connected Components Graph](images_gifs/SC_components_gif.gif)

![results](images_gifs/networkx_visuals)
