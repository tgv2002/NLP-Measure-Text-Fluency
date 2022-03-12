import json
import pandas as pd
import os
import sys

def create_report(folder_name):
    all_results = [f'{folder_name}_train_0_stats.json', f'{folder_name}_test_0_stats.json', f'{folder_name}_train_1_stats.json', 
                   f'{folder_name}_test_1_stats.json', f'{folder_name}_train_2_stats.json', f'{folder_name}_test_2_stats.json']
    names = ['Train (None)', 'Test (None)', 'Train (Random Over)', 'Test (Random Over)', 'Train (SMOTE)', 'Test (SMOTE)']
    data = []
    print(all_results)
    for j in range(len(all_results)):
        res = all_results[j]
        with open(folder_name + '/' + res, 'r') as f:
            d = json.load(f)
            title = names[j]
            accuracy = round(d["accuracy"], 3)
            f1 = round(d["macro avg"]["f1-score"], 3)
            recall = round(d["macro avg"]["recall"], 3)
            prec = round(d["macro avg"]["precision"], 3)
            data.append([title, accuracy, recall, prec, f1])
    df = pd.DataFrame(data, columns=['Title', 'Accuracy', 'Recall', 'Precision', 'F1-score'])
    df.to_csv(folder_name + '/stats.csv', index=False)
    with open(folder_name + '/stats.txt', 'w') as f:
        for row in data:
            s = ''
            for e in row:
                s += ' & ' + str(e)
            s += ' \\\\'
            print(s, file=f)
    
create_report(sys.argv[1])  
            
            