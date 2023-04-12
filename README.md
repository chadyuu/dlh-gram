Reproduction of GRAM (graph-based attention model)
=========================================

## Reference

Edward Choi, Mohammad Taha Bahadori, Le Song, Walter F Stewart, and Jimeng Sun. 2017. Gram: graph-based attention model for healthcare representation learning. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pages 787–795.

## Jupyter Notebook
- gram.ipynb
- RNN/rnn.ipynb

## Datasets

- ADMISSIONS.csv
- DIAGNOSES_ICD.csv
- ccs_multi_dx_tool_2015.csv


## Basic workflow for the heart failure prediction

```
$ python --version
Python 2.7.13
````

```
$ python process_mimic_hfs.py ADMISSIONS.csv DIAGNOSES_ICD.csv output/output
```
outputs in `ouput` directory:

- output.hfs
- output.seqs
- output.types

```
$ python build_trees.py ccs_multi_dx_tool_2015.csv output/output.seqs output/output.types output2/output2
```
outputs in `output2` directory:

- output2.level1.pk
- output2.level2.pk
- output2.level3.pk
- output2.level4.pk
- output2.level5.pk
- output2.seqs
- output2.types

```
$ python run_10times.py
```
outputs in `gram_hf` directory:

- gram_hf_{train_ratio}.test_set
- gram_hf_{train_ratio}_{epoch_id}.test_probs

```
$ cd auc
$ python --version
Python 3.10.1

$ python auc.py 
```

shows like

```
AUC: 0.69, train_ratio: 0.07
AUC: 0.69, train_ratio: 0.14
AUC: 0.69, train_ratio: 0.21
AUC: 0.70, train_ratio: 0.28
AUC: 0.71, train_ratio: 0.35
AUC: 0.72, train_ratio: 0.42
AUC: 0.72, train_ratio: 0.49
AUC: 0.72, train_ratio: 0.56
AUC: 0.73, train_ratio: 0.63
AUC: 0.73, train_ratio: 0.7
```

## results with 100 epochs

### gram_hf

AUC: 0.70, train_ratio: 0.07
AUC: 0.84, train_ratio: 0.14
AUC: 0.91, train_ratio: 0.21
AUC: 0.93, train_ratio: 0.28
AUC: 0.94, train_ratio: 0.35
AUC: 0.92, train_ratio: 0.42
AUC: 0.94, train_ratio: 0.49
AUC: 0.94, train_ratio: 0.56
AUC: 0.95, train_ratio: 0.63
AUC: 0.94, train_ratio: 0.7

### gram_hf_no_attention

AUC: 0.70, train_ratio: 0.07
AUC: 0.91, train_ratio: 0.14
AUC: 0.91, train_ratio: 0.21
AUC: 0.93, train_ratio: 0.28
AUC: 0.94, train_ratio: 0.35
AUC: 0.94, train_ratio: 0.42
AUC: 0.94, train_ratio: 0.49
AUC: 0.94, train_ratio: 0.56
AUC: 0.95, train_ratio: 0.63
AUC: 0.94, train_ratio: 0.7

### rnn.ipynb (NaiveRNN) for HF

AUC: 0.85, train_ratio: 0.07
AUC: 0.89, train_ratio: 0.14
AUC: 0.8953, train_ratio: 0.21
AUC: 0.9079, train_ratio: 0.28
AUC: 0.9176, train_ratio: 0.35
AUC: 0.9236, train_ratio: 0.42
AUC: 0.9231, train_ratio: 0.49
AUC: 0.9260, train_ratio: 0.56
AUC: 0.9210, train_ratio: 0.63
AUC: 0.9200, train_ratio: 0.70

## Hyperparameter tuning
```
python hyperparams_tunining.py # train the model
python auc/auc_hyperparam.py # calculate AUC
```

----------------------

## Basic workflows for the sequential diagnoses prediction

confirmed python2.7 (2.7.13) works.
not yet python3.x

#### Data preprocessing

```
$ python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv output
```

output

- output.3digitICD9.seqs
- output.3digitICD9.types
- output.dates
- output.pids
- output.seqs
- output.types

#### Build files that contain the ancestor information of each medical code

```
$ python build_trees.py ccs_multi_dx_tool_2015.csv output.seqs output.types output2
```
output
- output2.level1.pk
- output2.level2.pk
- output2.level3.pk
- output2.level4.pk
- output2.level5.pk
- output2.seqs
- output2.types


#### Run GRAM (train)
```
$ python gram.py output2.seqs output.3digitICD9.seqs output2 output3
```
output
- output3.*.npz


### Pretrain the code embedding
This is for GRAM+ and RNN+, so we do not need to pretrain the code embedding.
```
$ python create_glove_comap.py output2.seqs output2 output_pretrain
``` 
output
- cooccurrenceMa.pk
- output_pretrain2.0.npz
- output_pretrain2.1.npz
