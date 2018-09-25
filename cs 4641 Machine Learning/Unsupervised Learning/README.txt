Install
> Weka

	> Download and install at https://www.cs.waikato.ac.nz/ml/weka/downloading.html
	> it must be added to the classpath
		> export CLASSPATH=$CLASSPATH:/home/lemons/Documents/weka-3-8-2/weka.jar
	> The gui can be opened using java -jar weka.jar in the weka folder 
	> Once the gui is open go to tools->package manager then search for independent components and install the package

> pip install liac-arff

> pip install scikit-learn

Running the code
> Dimension Reduction (Done for you) 
	>open the weka explorer gui
	>In preprocess
		>open the dataset file
		>Use the correct filter
			> PCA = PrincipalComponents
			> ICA = IndependentComponents
				>Have to remove label attribute and set class to none
			> RP = RandomProjection
			>IG change to the select attribute tab
				> Select the InfoGainAttributeEval
				> allow weka to pink search method
				> Per the results select a threshold and remove attributes in the preprocessing tab
		> Save the end results in the dataset folder
	> Run convert_ICA.py to add labels back to ICA
Clustering
> java ClusteringDemo
> This will run all the cases and save the clustered data to the folders
Nerual Networks
> run NN.py
> this will run the neural nets and save the output
