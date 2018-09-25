import numpy as np
import pandas as pd
from pandas.plotting import table
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def plot_1var_graph(df, var, title, xlabel, ylabel, file_name="graph.png",height=4, width=10):
	fig, ax = plt.subplots()
	ax = df.plot(ax = ax, x = var)
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	fig.set_figheight(height)
	fig.set_figwidth(width)
	fig.savefig(file_name, bbox_inches='tight')
	plt.close()

def plot_2var_graph(df, var1, var2, y, title, xlabel, ylabel, file_name="graph.png", height=4, width=10):
	fig, ax = plt.subplots()
	for key, grp in df.groupby([var2]):
		ax = grp.plot(ax=ax, x = var1, y = y, label = key)
	ax.get_legend().set_title(var2)
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	fig.set_figheight(height)
	fig.set_figwidth(width)
	fig.savefig(file_name, bbox_inches='tight')
	plt.close()

def plot_NN_graph(df, var, title, xlabel, ylabel, file_name="graph.png"):
	fig, axs1 = plt.subplots(3,3, sharex="col", sharey="row")
	axs = []
	for row in axs1:
		for col in row:
			axs.append(col)
	i = 0
	for key, grp in df.groupby(["learning_rate_init", "momentum"]):
		ax = axs[i]
		ax = grp.plot(ax=ax, x = var, y = "training_acc", label = "Training")
		ax = grp.plot(ax=ax, x = var, y = "test_acc", label = "Test")
		ax.legend().set_visible(False)
		ax.set_title("l="+ str(key[0]) +", m=" + str(key[1]))
		i = i + 1
	fig.suptitle(title, y=1.08)
	for i in range(0,9):
		if i is 3:
			axs[3].set_ylabel(ylabel)
		elif i is 7:
			axs[7].set_xlabel(xlabel)
		else:
			axs[i].set_xlabel("")
	axs1.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=3)
	fig.tight_layout()

	fig.set_figheight(4)
	fig.set_figwidth(10)

	fig.savefig(file_name, bbox_inches='tight')
	plt.close()



datasets = [ "winequality-white" , "magic04"]
datatypes = ["knn", "DecisionTree", "Boost-max_depth_10", "Boost-max_depth_50", "SVM_linear", "SVM_rbf", "NN"]
for dataset in datasets:
	data = pd.read_csv(dataset+ "/"+ dataset + ".csv")
	if "train_time" in data:
		data = data.drop("train_time")
		data = data.drop("test_time")
	le = LabelEncoder()
	for col in data.columns.values:
		if data[col].dtypes == 'object':
			le.fit(data[col].values)
			data[col] = le.transform(data[col])
	data = (data-data.min())/(data.max() - data.min())
	fig, ax = plt.subplots()
	ax = data.hist(ax=ax)
	plt.suptitle(dataset + " Distribution", y=1.08)
	fig.tight_layout()
	fig.savefig(dataset+"/distribution.png", bbox_inches='tight')
	plt.close()


	datatype = "knn"
	data = pd.read_csv(dataset+ "/"+ dataset + "-" + datatype + ".csv", index_col=0)
	if "train_time" in data:
		data = data.drop("train_time")
		data = data.drop("test_time")
	max_sample_table = data.iloc[data.groupby("n_neighbors")["train_size"].agg(pd.Series.idxmax)]
	max_sample_table = max_sample_table.drop('train_size', 1)
	plot_1var_graph(max_sample_table, "n_neighbors",dataset+" KNN Accuracy over k values", "k", "Test Accuracy", file_name=dataset+"/"+datatype+"_max_sample.png")
	max_index = max_sample_table.cross_val_acc.idxmax()
	print(max_sample_table.ix[max_index])
	learning_curve_table = data[data['n_neighbors'] == max_sample_table.ix[max_index, 'n_neighbors']]
	learning_curve_table = learning_curve_table.drop('n_neighbors', 1)
	learning_curve_table.sort_values(by=["train_size"], inplace=True)
	plot_1var_graph(learning_curve_table, "train_size", dataset+" KNN Learning Curve", "training size", "Accuracy", file_name=dataset+"/"+datatype+"_learning_curve.png", height=4, width=5)


	datatype = "DecisionTree"
	data = pd.read_csv(dataset+ "/"+ dataset + "-" + datatype + ".csv", index_col=0)
	if "train_time" in data:
		data = data.drop("train_time")
		data = data.drop("test_time")
	data = data[data['criterion'] == "entropy"]
	data = data.reset_index(drop=True)
	max_sample_table = data.iloc[data.groupby(["criterion", "max_depth", "min_samples_leaf"])["train_size"].agg(pd.Series.idxmax)]
	max_sample_table = max_sample_table.drop('train_size', 1)
	plot_2var_graph(max_sample_table, "max_depth", "min_samples_leaf", "test_acc", dataset+" Decision Tree Accuracy over Max Depth", "Max Depth", "Test Accuracy", file_name=dataset+"/"+datatype+"_max_sample.png")
	max_index = max_sample_table.cross_val_acc.idxmax()
	print(max_sample_table.ix[max_index])
	learning_curve_table = data[data['max_depth'] == max_sample_table.ix[max_index, 'max_depth']]
	learning_curve_table = learning_curve_table[learning_curve_table['min_samples_leaf'] == max_sample_table.ix[max_index, 'min_samples_leaf']]
	learning_curve_table = learning_curve_table.drop('max_depth', 1)
	learning_curve_table = learning_curve_table.drop('min_samples_leaf', 1)
	learning_curve_table.sort_values(by=["train_size"], inplace=True)
	plot_1var_graph(learning_curve_table, "train_size", dataset+" Decision Tree Learning Curve", "training size", "Accuracy", file_name=dataset+"/"+datatype+"_learning_curve.png", height=4, width=5)

	datatype = "Boost-max_depth_10"
	data = pd.read_csv(dataset+ "/"+ dataset + "-" + datatype + ".csv", index_col=0)
	if "train_time" in data:
		data = data.drop("train_time")
		data = data.drop("test_time")
	data = data[data['algorithm'] == "SAMME"]
	data = data.reset_index(drop=True)
	max_sample_table = data.iloc[data.groupby(["algorithm", "n_estimators", "learning_rate"])["train_size"].agg(pd.Series.idxmax)]
	max_sample_table = max_sample_table.drop('train_size', 1)
	plot_2var_graph(max_sample_table, "n_estimators", "learning_rate", "test_acc", dataset+" Boost Max Depth 10 Accuracy over Number of Estimators", "Number of Estimators", "Test Accuracy", file_name=dataset+"/"+datatype+"_max_sample.png")
	max_index = max_sample_table.cross_val_acc.idxmax()
	print(max_sample_table.ix[max_index])
	learning_curve_table = data[data['n_estimators'] == max_sample_table.ix[max_index, 'n_estimators']]
	learning_curve_table = learning_curve_table[learning_curve_table['learning_rate'] == max_sample_table.ix[max_index, 'learning_rate']]
	learning_curve_table = learning_curve_table[learning_curve_table['algorithm'] == max_sample_table.ix[max_index, 'algorithm']]
	learning_curve_table = learning_curve_table.drop('algorithm', 1)
	learning_curve_table = learning_curve_table.drop('n_estimators', 1)
	learning_curve_table = learning_curve_table.drop('learning_rate', 1)
	learning_curve_table.sort_values(by=["train_size"], inplace=True)
	plot_1var_graph(learning_curve_table, "train_size", dataset+" Boost Max Depth 10 Learning Curve", "training size", "Accuracy", file_name=dataset+"/"+datatype+"_learning_curve.png", height=4, width=5)

	datatype = "SVM_linear"
	data = pd.read_csv(dataset+ "/"+ dataset + "-" + datatype + ".csv", index_col=0)
	if "train_time" in data:
		data = data.drop("train_time")
		data = data.drop("test_time")
	data = data[data['C'] <= 10]
	data = data[data['max_iter'] == 100]

	data = data.reset_index(drop=True)

	max_sample_table = data.iloc[data.groupby(["max_iter", "C"])["train_size"].agg(pd.Series.idxmax)]
	max_sample_table = max_sample_table.drop('train_size', 1)
	max_sample_table = max_sample_table.drop('max_iter', 1)
	plot_1var_graph(max_sample_table, "C",dataset+" SVM linear Accuracy over C values", "C", "Test Accuracy", file_name=dataset+"/"+datatype+"_max_sample.png")
	max_index = max_sample_table.cross_val_acc.idxmax()
	print(max_sample_table.ix[max_index])
	learning_curve_table = data[data['kernel'] == max_sample_table.ix[max_index, 'kernel']]
	learning_curve_table = learning_curve_table[learning_curve_table['C'] == max_sample_table.ix[max_index, 'C']]
	learning_curve_table = learning_curve_table.drop('C', 1)
	learning_curve_table = learning_curve_table.drop('kernel', 1)
	learning_curve_table = learning_curve_table.drop('max_iter', 1)
	learning_curve_table.sort_values(by=["train_size"], inplace=True)
	plot_1var_graph(learning_curve_table, "train_size", dataset+" SVM Linear Learning Curve", "training size", "Accuracy", file_name=dataset+"/"+datatype+"_learning_curve.png", height=4, width=5)

	datatype = "SVM_rbf"
	data = pd.read_csv(dataset+ "/"+ dataset + "-" + datatype + ".csv", index_col=0)
	if "train_time" in data:
		data = data.drop("train_time")
		data = data.drop("test_time")
	data = data[data['max_iter'] == 1000]
	data = data[data['C'] <= 10]
	data = data.reset_index(drop=True)
	max_sample_table = data.iloc[data.groupby(["max_iter", "gamma", "C"])["train_size"].agg(pd.Series.idxmax)]
	max_sample_table = max_sample_table.drop('train_size', 1)
	plot_2var_graph(max_sample_table, "C", "gamma", "test_acc", dataset+" SVM RBF Accuracy over C", "C", "Test Accuracy", file_name=dataset+"/"+datatype+"_max_sample.png")
	max_index = max_sample_table.cross_val_acc.idxmax()
	print(max_sample_table.ix[max_index])
	learning_curve_table = data[data['kernel'] == max_sample_table.ix[max_index, 'kernel']]
	learning_curve_table = learning_curve_table[learning_curve_table['max_iter'] == max_sample_table.ix[max_index, 'max_iter']]
	learning_curve_table = learning_curve_table[learning_curve_table['gamma'] == max_sample_table.ix[max_index, 'gamma']]
	learning_curve_table = learning_curve_table[learning_curve_table['C'] == max_sample_table.ix[max_index, 'C']]
	learning_curve_table = learning_curve_table.drop('C', 1)
	learning_curve_table = learning_curve_table.drop('gamma', 1)
	learning_curve_table = learning_curve_table.drop('kernel', 1)
	learning_curve_table = learning_curve_table.drop('max_iter', 1)
	learning_curve_table.sort_values(by=["train_size"], inplace=True)
	plot_1var_graph(learning_curve_table, "train_size", dataset+" SVM RBF Learning Curve", "training size", "Accuracy", file_name=dataset+"/"+datatype+"_learning_curve.png", height=4, width=5)

dataset= "winequality-white"
datatype = "NN"
data = pd.read_csv(dataset+ "/"+ dataset + "-" + datatype + ".csv", index_col=0)
if "train_time" in data:
	data = data.drop("train_time")
	data = data.drop("test_time")
data = data[data['learning_rate_init'] < 1]
data = data[data['momentum'] <= 0.9]
data = data[data['momentum'] >= 0.2]
data = data[data['momentum'] != 0.7]
data = data[data['hidden_layer_sizes'].isin(['(13, 13)', '(130, 130, 130, 130)'])]

data = data.reset_index(drop=True)
max_index = data.cross_val_acc.idxmax()
print(data.ix[max_index])
learning_curve_table = data[data['hidden_layer_sizes'] == '(13, 13)']
learning_curve_table = learning_curve_table[learning_curve_table['solver'] == "sgd"]
learning_curve_table = learning_curve_table[learning_curve_table['train_size'] == learning_curve_table['train_size'].max()]
learning_curve_table.sort_values(by=["max_iter", "momentum", "learning_rate_init", "train_size"], inplace=True)
learning_curve_table = learning_curve_table.drop('train_size', 1)
learning_curve_table = learning_curve_table.drop('hidden_layer_sizes', 1)
learning_curve_table = learning_curve_table.drop('solver', 1)
plot_NN_graph(learning_curve_table, "max_iter", dataset+" NN Learning Curve (13,13)", "Epochs", "Accuracy", file_name=dataset+"/"+datatype+"_learning_curve.png")

learning_curve_table = data[data['hidden_layer_sizes'] == '(130, 130, 130, 130)']
learning_curve_table = learning_curve_table[learning_curve_table['solver'] == "sgd"]
learning_curve_table = learning_curve_table[learning_curve_table['train_size'] == learning_curve_table['train_size'].max()]
learning_curve_table.sort_values(by=["max_iter", "momentum", "learning_rate_init", "train_size"], inplace=True)
learning_curve_table = learning_curve_table.drop('train_size', 1)
learning_curve_table = learning_curve_table.drop('hidden_layer_sizes', 1)
learning_curve_table = learning_curve_table.drop('solver', 1)
plot_NN_graph(learning_curve_table, "max_iter", dataset+" NN Learning Curve (130,130,130,130)", "Epochs", "Accuracy", file_name=dataset+"/"+datatype+"2_learning_curve.png")

dataset= "magic04"
datatype = "NN"
data = pd.read_csv(dataset+ "/"+ dataset + "-" + datatype + ".csv", index_col=0)
if "train_time" in data:
	data = data.drop("train_time")
	data = data.drop("test_time")
data = data[data['learning_rate_init'] < 1]
data = data[data['momentum'].isin([0.2, 0.5, 0.9])]
data = data[data['hidden_layer_sizes'].isin(['(100,)', '(130, 130)'])]
data = data.reset_index(drop=True)
max_index = data.cross_val_acc.idxmax()
print(data.ix[max_index])
learning_curve_table = data[data['hidden_layer_sizes'] == '(100,)']
learning_curve_table = learning_curve_table[learning_curve_table['solver'] == "sgd"]
learning_curve_table = learning_curve_table[learning_curve_table['train_size'] == learning_curve_table['train_size'].max()]
learning_curve_table.sort_values(by=["max_iter", "momentum", "learning_rate_init", "train_size"], inplace=True)
learning_curve_table = learning_curve_table.drop('train_size', 1)
learning_curve_table = learning_curve_table.drop('hidden_layer_sizes', 1)
learning_curve_table = learning_curve_table.drop('solver', 1)
plot_NN_graph(learning_curve_table, "max_iter", dataset+" NN Learning Curve (100)", "Epochs", "Accuracy", file_name=dataset+"/"+datatype+"_learning_curve.png")

learning_curve_table = data[data['hidden_layer_sizes'] == '(130, 130)']
learning_curve_table = learning_curve_table[learning_curve_table['solver'] == "sgd"]
learning_curve_table = learning_curve_table[learning_curve_table['train_size'] == learning_curve_table['train_size'].max()]
learning_curve_table.sort_values(by=["max_iter", "momentum", "learning_rate_init", "train_size"], inplace=True)
learning_curve_table = learning_curve_table.drop('train_size', 1)
learning_curve_table = learning_curve_table.drop('hidden_layer_sizes', 1)
learning_curve_table = learning_curve_table.drop('solver', 1)
plot_NN_graph(learning_curve_table, "max_iter", dataset+" NN Learning Curve (130,130)", "Epochs", "Accuracy", file_name=dataset+"/"+datatype+"2_learning_curve.png")



