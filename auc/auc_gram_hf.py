import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

'''
# (test_set_x, test_set_y)
print(test_data[0])
print(len(test_data[0])) # 1507
print(test_data[1])
print(len(test_data[1])) # 1507
'''

dir = '../gram_hf'

results = [(0.07,4),
           (0.14,51),
           (0.21,84),
           (0.28,60),
           (0.35,39),
           (0.42,28),
           (0.49,38),
           (0.56,40),
           (0.63,39),
           (0.7,46),
           ]

for train_ratio, epoch in results:
    with open(f'{dir}/gram_hf_{round(train_ratio,2)}.test_set', 'rb') as f:
        test_data = pickle.load(f)
    with open(f'{dir}/gram_hf_{round(train_ratio,2)}_{epoch}.test_probs', 'rb') as f:
        y_pred_probs = pickle.load(f, encoding='latin1')
    
    y_true_labels = test_data[1]
    auc = roc_auc_score(y_true_labels, y_pred_probs)

    print(f"AUC: {auc:.2f}, train_ratio: {round(train_ratio,2)}")