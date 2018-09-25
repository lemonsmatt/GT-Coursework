#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=ABAGAIL.jar:$CLASSPATH
mkdir -p out plots



#Count ones
echo "count ones"
jython countones.py

#Traveling Salesman
echo "Traveling Salesman"
jython traveling_salesman.py

#Knapsack
echo "Knapsack"
jython knapsack.py

#NN
echo "NN"
jython NN.py

#Plotting
echo "Plotting"
python analysis_data.py


