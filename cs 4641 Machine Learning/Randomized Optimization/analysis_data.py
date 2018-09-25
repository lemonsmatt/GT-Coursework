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

def plot_NN_graph(df, title, file_name="graph.png"):
    fig, axs1 = plt.subplots(1,2, sharex="col", sharey="row")
    axs = []
    for row in axs1:
        axs.append(row)
    axs[0].set_title("error vs iterations")
    axs[0].set_xlabel("iterations")
    axs[0].set_ylabel("error")
    axs[1].set_title("error vs runtime")
    axs[1].set_xlabel("runtime")
    axs[1].set_ylabel("")
    ax = axs[0]

    ax = df.plot(ax=ax, x = "iterations", y = "training_error", label = "Training")
    ax = df.plot(ax=ax, x = "iterations", y = "test_error", label = "Test")
    ax.legend().set_visible(False)

    ax = axs[1]

    ax = df.plot(ax=ax, x = "run_time", y = "training_error", label = "Training")
    ax = df.plot(ax=ax, x = "run_time", y = "test_error", label = "Test")
    ax.legend().set_visible(False)

    fig.suptitle(title, y=1.08)
    axs1.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1, -0.2), ncol=3)
    fig.tight_layout()

    fig.set_figheight(4)
    fig.set_figwidth(10)

    fig.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_settings(df, setting1, setting2, title, file_name="graph.png"):
    df2 = df.sort_values(["N", "iterations"])
    df = pd.DataFrame()
    for key1, grp1 in df2.groupby(["N", setting1, setting2]):
        max_val = 0
        for key, grp in grp1.groupby(["iterations"]):
            max_val = max(grp["value"].max(), max_val)
            grp["value"] = max_val
            df = pd.concat([df, grp])
    fig, axs1 = plt.subplots(1,2, sharex="col", sharey="row")
    axs = []
    for row in axs1:
        axs.append(row)
    axs[0].set_title("Value vs iterations")
    axs[0].set_xlabel("iterations")
    axs[0].set_ylabel("Value")
    axs[1].set_title("Value vs runtime")
    axs[1].set_xlabel("runtime")
    axs[1].set_ylabel("")
    for key, grp in df.groupby([setting1, setting2]):
        ax = axs[0]
        ax = grp.plot(ax=ax, x = "iterations", y = "value", label = setting1 + "="+ str(key[0]) + ", " + setting2 + "=" + str(key[1]))
        ax.legend().set_visible(False)
        ax = axs[1]
        ax = grp.plot(ax=ax, x = "run_time", y = "value", label = setting1 + "="+ str(key[0]) + ", " + setting2 + "=" + str(key[1]))
        ax.legend().set_visible(False)
    fig.suptitle(title, y=1.08)

    axs1.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1, -0.2), ncol=3)
    fig.tight_layout()

    fig.set_figheight(4)
    fig.set_figwidth(10)

    fig.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_select(df, title, file_name="graph.png", N_query = "N == 80"):
    df2 = df.sort_values(["N", "iterations"])
    df = pd.DataFrame()
    for key1, grp1 in df2.groupby(["N"]):
        max_val = {"GA": 0, "SA": 0, "MIMIC": 0, "RHC": 0}
        for key, grp in grp1.groupby(["algo", "iterations"]):
            max_val[key[0]] = max(grp["value"].max(), max_val[key[0]])
            grp["value"] = max_val[key[0]]
            df = pd.concat([df, grp])
    df_N = df.query("iterations == 10000 or (iterations == 5000 and algo == 'MIMIC')")
    df_iter = df.query(N_query)
    df_run = df.query(N_query)


    fig, axs1 = plt.subplots(1,3, sharex="col", sharey="row")
    axs = []
    for row in axs1:
        axs.append(row)
    axs[0].set_title("Value vs Problem size")
    axs[0].set_xlabel("Problem Size")
    axs[0].set_ylabel("")
    axs[1].set_title("Value vs iterations")
    axs[1].set_xlabel("iterations")
    axs[1].set_ylabel("")
    axs[2].set_title("Value vs runtime")
    axs[2].set_xlabel("runtime")
    axs[2].set_ylabel("")

    for key, grp in df_N.groupby(["algo"]):
        print(key)
        ax = axs[0]
        ax = grp.plot(ax=ax, x = "N", y = "value", label = str(key))
        ax.legend().set_visible(False)

    for key, grp in df_iter.groupby(["algo"]):
        ax = axs[1]
        ax = grp.plot(ax=ax, x = "iterations", y = "value", label = str(key))
        ax.legend().set_visible(False)

    for key, grp in df_run.groupby(["algo"]):
        ax = axs[2]
        ax = grp.plot(ax=ax, x = "run_time", y = "value", label = str(key))
        ax.legend().set_visible(False)

    fig.suptitle(title, y=1.08)

    axs1.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1, -0.2), ncol=4)
    fig.tight_layout()

    fig.set_figheight(4)
    fig.set_figwidth(10)

    fig.savefig(file_name, bbox_inches='tight')
    plt.close()


problems = ["fourpeaks", "countones", "knapsack", "traveling_salesman"]
settings = [[[1E5, 0.8],[300,1.0], [200,0.2]], [[1E10, 0.95], [300, 1.0], [100,0.1]], [[1E5, 0.8] ,[200,0.5], [50, 0.1]], [[1E5,0.6], [20,0.5], [200,0.1]]]
for problem,setting in zip(problems,settings):
    N_val = 80
    if problem is "traveling_salesman":
        N_val = 40
    df_select = pd.DataFrame()

    algo = "RHC"
    label = ["N", "iterations", "run_time", "call_count", "value"]
    data = pd.read_csv("out" + "/" + problem + "_"+ algo + ".csv")
    data.columns = label
    data["algo"] = algo
    df_select = pd.concat([df_select, data[["N", "iterations", "run_time", "call_count", "value", "algo"]]])


    algo = "SA"
    label = ["N", "iterations", "run_time", "call_count", "value" , "t", "cooling"]
    data = pd.read_csv("out" + "/" + problem + "_"+ algo + ".csv")
    data.columns = label

    df_setting = data.loc[data["N"] == N_val]
    plot_settings(df_setting, "t", "cooling", problem + " " + algo + " Settings" ,"plots/" + problem+"_" +algo+"_setting.png")
    data["algo"] = algo
    df_select = pd.concat([df_select, data.loc[data["t"] == setting[0][0]].loc[data["cooling"] == setting[0][1]][["N", "iterations", "run_time", "call_count", "value", "algo"]]])

    algo = "GA"
    label = ["N", "iterations", "run_time", "call_count", "value", "populationSize", "toMate", "toMutate", "crossover"]
    data = pd.read_csv("out" + "/" + problem + "_"+ algo + ".csv")
    data.columns = label
    data["algo"] = algo
    data = data.loc[data["populationSize"] != 400].loc[data["populationSize"] !=2000 ]
    df_setting = data.loc[data["N"] == N_val]
    plot_settings(df_setting, "populationSize", "toMate", problem + " " + algo + " Settings" ,"plots/" + problem+"_" +algo+"_setting.png")
    df_select = pd.concat([df_select, data.loc[data["populationSize"] == setting[1][0]].loc[data["toMate"] == setting[1][1]][["N", "iterations", "run_time", "call_count", "value", "algo"]]])


    algo = "MIMIC"
    label = ["N", "iterations", "run_time", "call_count", "value", "samples", "tokeep"]
    data = pd.read_csv("out" + "/" + problem + "_"+ algo + ".csv")
    data.columns = label
    data["algo"] = algo
    data = data.loc[data["samples"] != 500]

    df_setting = data.loc[data["N"] == N_val]
    plot_settings(df_setting, "samples", "tokeep", problem + " " + algo + " Settings" ,"plots/" + problem+"_" +algo+"_setting.png")
    df_select = pd.concat([df_select, data.loc[data["samples"] == setting[2][0]].loc[data["tokeep"] == setting[2][1]][["N", "iterations", "run_time", "call_count", "value", "algo"]]])

    if problem == "traveling_salesman":
        plot_select(df_select, problem, file_name= "plots/" + problem  + "_select.png", N_query="N == 40")
    else:
        plot_select(df_select, problem, file_name= "plots/" + problem  + "_select.png")

algorithms = ["GA", "SA", "RHC"]
for algo in algorithms:
    label = ["iterations", "run_time", "training_error", "test_error"]
    data = pd.read_csv("out" + "/NN_"+ algo + ".csv")
    data.columns = label
    plot_NN_graph(data, algo, "plots/NN_" + algo + ".png")
