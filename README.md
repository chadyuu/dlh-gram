Reproduction of GRAM (graph-based attention model) for the heart failure prediction
=========================================

## Reference

Edward Choi, Mohammad Taha Bahadori, Le Song, Walter F Stewart, and Jimeng Sun. 2017. Gram: graph-based attention model for healthcare representation learning. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pages 787–795.

## Requirements

### Train the GRAM model

For this step we used Python [2.7.13](.python-version) with Pyenv.

To install requirements:
```
pip install -r requirements.txt
```

### Calculate AUC

For this step we used Python [3.10.1](auc/.python-version) with Pyenv.

To install requirements:
```
cd auc
pip install -r requirements.txt
```

## Datasets

You can download the following necessary datasets from https://physionet.org/content/mimiciii/1.4/ after obtaining the access privilege.

- ADMISSIONS.csv
- DIAGNOSES_ICD.csv
- ccs_multi_dx_tool_2015.csv


## Preprocess the data 

```
$ python process_mimic_hfs.py ADMISSIONS.csv DIAGNOSES_ICD.csv output/output
```
outputs in `ouput` directory:

- output.hfs
- output.seqs
- output.types

```
# Build files that contain the ancestor information of each medical code
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

## Train the GRAM model and generate predictions

```
$ python run_10times.py
```
outputs in `gram_hf` directory:

- gram_hf_{train_ratio}.test_set
- gram_hf_{train_ratio}_{epoch_id}.test_probs


As an ablation, we removed the attention mechanism:

```
$ python run_10times_no_attention.py
```
outputs in `gram_hf_no_attention` directory:

- gram_hf_{train_ratio}.test_set
- gram_hf_{train_ratio}_{epoch_id}.test_probs


## Evaluate predictions
```
$ cd auc
$ python auc.py 
```

## Results

### gram_hf

- AUC: 0.695, train_ratio: 0.07
- AUC: 0.844, train_ratio: 0.14
- AUC: 0.909, train_ratio: 0.21
- AUC: 0.926, train_ratio: 0.28
- AUC: 0.936, train_ratio: 0.35
- AUC: 0.923, train_ratio: 0.42
- AUC: 0.937, train_ratio: 0.49
- AUC: 0.940, train_ratio: 0.56
- AUC: 0.947, train_ratio: 0.63
- AUC: 0.944, train_ratio: 0.7

### gram_hf_no_attention

- AUC: 0.698, train_ratio: 0.07
- AUC: 0.911, train_ratio: 0.14
- AUC: 0.912, train_ratio: 0.21
- AUC: 0.929, train_ratio: 0.28
- AUC: 0.941, train_ratio: 0.35
- AUC: 0.938, train_ratio: 0.42
- AUC: 0.944, train_ratio: 0.49
- AUC: 0.943, train_ratio: 0.56
- AUC: 0.945, train_ratio: 0.63
- AUC: 0.943, train_ratio: 0.7


## Hyperparameter tuning

```
# Train the model, output results in the `hyperparam` directory.
python hyperparams_tunining.py 

# Calculate AUC
cd auc
python auc_hyperparam.py
```
Here are the calculated AUC:

- AUC: 0.878 for default_0.7.test_set
- AUC: 0.855 for embDimSize100_0.7.test_set
- AUC: 0.890 for embDimSize300_0.7.test_set
- AUC: 0.894 for embDimSize400_0.7.test_set
- AUC: 0.893 for embDimSize500_0.7.test_set
- AUC: 0.900 for hiddenDimSize200_0.7.test_set
- AUC: 0.908 for hiddenDimSize300_0.7.test_set
- AUC: 0.923 for hiddenDimSize400_0.7.test_set
- AUC: 0.917 for hiddenDimSize500_0.7.test_set
- AUC: 0.887 for attentionDimSize200_0.7.test_set
- AUC: 0.877 for attentionDimSize300_0.7.test_set
- AUC: 0.861 for attentionDimSize400_0.7.test_set
- AUC: 0.873 for attentionDimSize500_0.7.test_set
- AUC: 0.884 for L20.0001_0.7.test_set
- AUC: 0.872 for L20.01_0.7.test_set
- AUC: 0.864 for L20.1_0.7.test_set
- AUC: 0.484 for dropoutRate0.0_0.7.test_set
- AUC: 0.869 for dropoutRate0.2_0.7.test_set
- AUC: 0.873 for dropoutRate0.4_0.7.test_set
- AUC: 0.889 for dropoutRate0.8_0.7.test_set
- AUC: 0.921 for embDimSize500_hiddenDimSize500_0.7.test_set
- AUC: 0.921 for best_0.7.test_set

## Comparison with NaiveRNN

See [RNN/rnn.ipynb](RNN/rnn.ipynb).

Here are the AUC for NaiveRNN.

- AUC: 0.85, train_ratio: 0.07
- AUC: 0.89, train_ratio: 0.14
- AUC: 0.8953, train_ratio: 0.21
- AUC: 0.9079, train_ratio: 0.28
- AUC: 0.9176, train_ratio: 0.35
- AUC: 0.9236, train_ratio: 0.42
- AUC: 0.9231, train_ratio: 0.49
- AUC: 0.9260, train_ratio: 0.56
- AUC: 0.9210, train_ratio: 0.63
- AUC: 0.9200, train_ratio: 0.70


## Appendix
You can see a sample workflow in [gram.ipynb](gram.ipynb).


## Contributing
If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at yutaron2@illinois.edu or open an issue on this GitHub repository.
