import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt
import os

FIG_PATH = '../publication/figures'
TABLE_PATH = '../publication/tables'

# Change to fit dataset and model
DATASET = 'measurements'
START_FILTER = 3950

# DATASET = 'powertrain'
# START_FILTER = 150025

# MODEL_FILTER = 'llama3.1:8b-instruct-fp16'
MODEL_FILTER = 'mistral'

RESULTS_PATH = f"../results/{DATASET}/"


def evaluate(results, window=100):
    true_anomalies = int(window / 2)
    df = pd.read_csv(results)
    #print(df.iloc[true_anomalies - 10: true_anomalies + 10].reason)
    y_pred = df.get("anomaly").astype(int).to_numpy()
    y_train1 = np.zeros((true_anomalies)).astype(int)
    y_train2 = np.ones((y_pred.shape[0] - true_anomalies)).astype(int)
    y_train = np.concatenate((y_train1, y_train2))

    # cm = confusion_matrix(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    # print("F1 score: ", f1)

    # plt.matshow(cm)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()
    return f1, precision, recall

def data_from_filename(filename):
    search_strategy = filename.split('_')[5]
    num_of_results = filename.split('_')[6]
    start = int(filename.split('_')[2])
    end = int(filename.split('_')[4])
    model = filename.split('_')[7].replace('.csv', '')
    return search_strategy, num_of_results, model, start, end

def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')

def create_latex_table(results, label, caption, filename):
    styler = results.style.format_index(escape="latex", axis=1).format_index(escape="latex", axis=0).format(precision=2)
    slice = ['F1']
    styler.apply(highlight_max, props="font-weight: bold; color:black; textit:--rwrap; textbf:--rwrap;", axis=0, subset=slice)
    table = styler.to_latex(
        caption=f"{caption}",
        label=f"tab:{label}",
        convert_css=True,
        position_float="centering",
        multicol_align="|c|",
        hrules=True,
        position="htbp",
        environment="sidewaystable",
    )

    with open(f"{TABLE_PATH}/{filename}", "w") as f:
        f.write(table)
        f.close()

    print(f"Table saved to {TABLE_PATH}/{filename}")

results = pd.DataFrame()
for filename in os.listdir(RESULTS_PATH):
    if filename.endswith(".csv"):
        search_strategy, num_of_results, model, start, end = data_from_filename(filename)

        print(f"model: {model}, search_strategy: {search_strategy}, num_of_results: {num_of_results}, start: {start}, end: {end}")
        window = int(int(end) - int(start))
        if model == MODEL_FILTER and start == START_FILTER:
            f1, precision, recall = evaluate(RESULTS_PATH + filename, window)
            results = pd.concat([results, pd.DataFrame({
                'Model': [model],
                #'dataset': "battery",
                'Strategy': [search_strategy],
                '# Results': [num_of_results],
                'F1': [f1], 
                'Precision': [precision], 
                'Recall': [recall], 
                }, index=[f"{model}_{num_of_results}_{search_strategy}"])])

            results.sort_values(by=['Model', '# Results', 'Strategy'], inplace=True)

            plot = results.plot.bar(rot=45, title=f"F1-score, Precision, and Recall for {model} and {window} test samples", legend=True)
            plot.get_figure().savefig(f"{FIG_PATH}/{DATASET}-{model}-results.pdf", format='pdf', bbox_inches='tight')
            plot.get_figure().savefig(f"{FIG_PATH}/{DATASET}-{model}-results.png", format='png', bbox_inches='tight')
            #plt.show()

            create_latex_table(results, 
                   label=f"{DATASET}-{model}-{num_of_results}-results",
                   caption=f"Results for dataset {DATASET} and {model} with context window of size{num_of_results}. Performance of Large Language Models on Anomaly Detection tested on 100 test samples.",
                   filename=f"{DATASET}-{model}-results.tex")