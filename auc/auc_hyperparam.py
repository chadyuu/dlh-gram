import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

files = [
    ('default_0.7_9.test_probs', 'default_0.7.test_set'),
    
    ('embDimSize100_0.7_9.test_probs', 'embDimSize100_0.7.test_set'),
    ('embDimSize300_0.7_8.test_probs', 'embDimSize300_0.7.test_set'),
    ('embDimSize400_0.7_9.test_probs', 'embDimSize400_0.7.test_set'),
    ('embDimSize500_0.7_9.test_probs', 'embDimSize500_0.7.test_set'),
    
    ('hiddenDimSize200_0.7_9.test_probs', 'hiddenDimSize200_0.7.test_set'),
    ('hiddenDimSize300_0.7_9.test_probs', 'hiddenDimSize300_0.7.test_set'),
    ('hiddenDimSize400_0.7_9.test_probs', 'hiddenDimSize400_0.7.test_set'),
    ('hiddenDimSize500_0.7_9.test_probs', 'hiddenDimSize500_0.7.test_set'),
    
    ('attentionDimSize200_0.7_9.test_probs', 'attentionDimSize200_0.7.test_set'),
    ('attentionDimSize300_0.7_9.test_probs', 'attentionDimSize300_0.7.test_set'),
    ('attentionDimSize400_0.7_9.test_probs', 'attentionDimSize400_0.7.test_set'),
    ('attentionDimSize500_0.7_9.test_probs', 'attentionDimSize500_0.7.test_set'),
    
    ('L20.0001_0.7_9.test_probs', 'L20.0001_0.7.test_set'),
    ('L20.01_0.7_9.test_probs', 'L20.01_0.7.test_set'),
    ('L20.1_0.7_9.test_probs', 'L20.1_0.7.test_set'),

    ('dropoutRate0.0_0.7_9.test_probs', 'dropoutRate0.0_0.7.test_set'),
    ('dropoutRate0.2_0.7_8.test_probs', 'dropoutRate0.2_0.7.test_set'),
    ('dropoutRate0.4_0.7_9.test_probs', 'dropoutRate0.4_0.7.test_set'),
    ('dropoutRate0.8_0.7_9.test_probs', 'dropoutRate0.8_0.7.test_set'),

    # possibly best
    ('embDimSize500_hiddenDimSize500_0.7_9.test_probs', 'embDimSize500_hiddenDimSize500_0.7.test_set'),
    ('best_0.7_9.test_probs', 'best_0.7.test_set'),
]

for log, test_set in files:
    with open(f'../hyperparam/{test_set}', 'rb') as f:
        test_data = pickle.load(f)
    with open(f'../hyperparam/{log}', 'rb') as f:
        y_pred_probs = pickle.load(f, encoding='latin1')
    
    y_true_labels = test_data[1]
    for i, y in enumerate(y_true_labels):
        y_true_labels[i] = y > 0.5
    auc = roc_auc_score(y_true_labels, y_pred_probs)

    print(f"AUC: {auc:.3f} for {test_set}")