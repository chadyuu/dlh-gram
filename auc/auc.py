import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

with open('../gram_hf.test_set', 'rb') as f:
    test_data = pickle.load(f)

'''
# (test_set_x, test_set_y)
print(test_data[0])
print(len(test_data[0])) # 1507
print(test_data[1])
print(len(test_data[1])) # 1507
'''
for train_ratio in np.arange(0.07,0.77,0.07):
    file_name= f'../gram_hf_{round(train_ratio,2)}_0.test_probs'
    
    with open(file_name, 'rb') as f:
        y_pred_probs = pickle.load(f, encoding='latin1')
    
    y_true_labels = test_data[1]
    auc = roc_auc_score(y_true_labels, y_pred_probs)

    print(f"AUC: {auc:.2f}, train_ratio: {round(train_ratio,2)}")