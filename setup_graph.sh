export MODULEPATH=/project2/msca/ivy2/software2/modules:$MODULEPATH

module load spark/2.4.4

alias jnc="jupyter notebook --ip=`hostname -i` --no-browser"
alias jnl="jupyter notebook --ip=`hostname -s`.rcc.uchicago.edu --no-browser"

#PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --ip=$h" pyspark

hc=`hostname -i`
hl=`hostname -s`.rcc.uchicago.edu
alias jncs="PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS=\"notebook --no-browser --ip=$hc\" pyspark --packages graphframes:graphframes:0.6.0-spark2.4.4-s_2.11"
alias jncs1="PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS=\"notebook --no-browser --ip=$hc\" pyspark --executor-memory=4G --driver-memory=8G --packages graphframes:graphframes:0.6.0-spark2.4.4-s_2.11"
alias jnls="PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS=\"notebook --no-browser --ip=$hl\" pyspark --packages graphframes:graphframes:0.6.0-spark2.4.4-s_2.11"
alias jnls1="PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS=\"notebook --no-browser --ip=$hl\" pyspark --executor-memory=4G --driver-memory=8G --packages graphframes:graphframes:0.6.0-spark2.4.4-s_2.11"
