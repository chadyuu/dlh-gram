GRAM
=========================================

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

### rnn.ipynb (NaiveRNN)

AUC: 0.85, train_ratio: 0.07
AUC: 0.89, train_ratio: 0.14
AUC: 0.8953, train_ratio: 0.21
AUC: 0.9079, train_ratio: 0.28
AUC: 0.9176, train_ratio: 0.35
AUC: 0.9236, train_ratio: 0.42
AUC: 0.9231, train_ratio: 0.49

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


*Have not run the following code.
``` 
$ python glove.py cooccurrenceMap.pk output2 output_pretrain2
$ python gram.py <seqs file> <3digitICD9.seqs file> <tree file prefix> <output path> --embed_file <embedding path> --embed_size <embedding dimension>
``` 

=========================================

GRAM is a prediction framework that can use the domain knowledge in the form of directed acyclic graph (DAG).
Domain knowedge is incorporated in the training process using the [attention mechanism](https://arxiv.org/abs/1409.0473). 
By introducing well established knoweldge into the training process, we can learn high quality representations of medical concepts that lead to more accurate predictions. 
The prediction task could take any form such as static prediction, sequence classification, or sequential prediction.

**t-SNE scatterplot of medical concepts trained with the combination of RNN and Multi-level Clincial Classification Software for ICD9** (The color of the dots represent the most general description of ICD9 diagnosis codes)
![tsne](http://www.cc.gatech.edu/~echoi48/images/gram_tsne.png "t-SNE scatterplot of medical concepts trained with the combination of RNN and Multi-level Clincial Classification Software for ICD9")

#### Relevant Publications

GRAM implements the algorithm introduced in the following [paper](https://arxiv.org/abs/1611.07012):

	GRAM: Graph-based Attention Model for Healthcare Representation Learning
	Edward Choi, Mohammad Taha Bahadori, Le Song, Walter F. Stewart, Jimeng Sun  
	Knowledge Discovery and Data Mining (KDD) 2017

#### Code Description

The current code trains an RNN ([Gated Recurrent Units](https://arxiv.org/abs/1406.1078)) to predict, at each timestep (i.e. visit), the diagnosis codes occurring in the next visit.
This is denoted as *Sequential Diagnoses Prediction* in the paper. 
In the future, we will relases another version for making a single prediction for the entire visit sequence. (e.g. Predict the onset of heart failure given the visit record)

Note that the current code uses [Multi-level Clinical Classification Software for ICD-9-CM](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp) as the domain knowledge.
We will release the one that uses ICD9 Diagnosis Hierarchy in the future.
	
#### Running GRAM

**STEP 1: Installation**  

1. Install [python](https://www.python.org/), [Theano](http://deeplearning.net/software/theano/index.html). We use Python 2.7, Theano 0.8.2. Theano can be easily installed in Ubuntu as suggested [here](http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu)

2. If you plan to use GPU computation, install [CUDA](https://developer.nvidia.com/cuda-downloads)

3. Download/clone the GRAM code  

**STEP 2: Fastest way to test GRAM with MIMIC-III**  

This step describes how to run, with minimum number of steps, GRAM for predicting future diagnosis codes using MIMIC-III. 

0. You will first need to request access for [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/), a publicly avaiable electronic health records collected from ICU patients over 11 years. 

1. You can use "process_mimic.py" to process MIMIC-III dataset and generate a suitable training dataset for GRAM. 
Place the script to the same location where the MIMIC-III CSV files are located, and run the script. 
Instructions are described inside the script. 

2. Use "build_trees.py" to build files that contain the ancestor information of each medical code. 
This requires "ccs_multi_dx_tool_2015.csv" (Multi-level CCS for ICD9), which can be downloaded from 
[here](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip).
Running this script will re-map integer codes assigned to all medical codes.
Therefore you also need the ".seqs" file and the ".types" file created by process_mimc.py.
The execution command is `python build_trees.py ccs_multi_dx_tool_2015.csv <seqs file> <types file> <output path>`. 
This will build five files that have ".level#.pk" as the suffix.
This will replace the old ".seqs" and ".types" files with the correct ones.
(Tian Bai, a PhD student from Temple University found out there was a problem with the re-mapping issue, which is now fixed. Thanks Tian!)

3. Run GRAM using the ".seqs" file generated by build_trees.py. 
The ".seqs" file contains the sequence of visits for each patient. Each visit consists of multiple diagnosis codes.
Instead of using the same ".seqs" file as both the training feature and the training label, 
we recommend using ".3digitICD9.seqs" file, which is also generated by process_mimic.py, as the training label for better performance and eaiser analysis.
The command is `python gram.py <seqs file> <3digitICD9.seqs file> <tree file prefix> <output path>`. 

**STEP 3: How to pretrain the code embedding**

For sequential diagnoses prediction, it is very effective to pretrain the code embeddings with some co-occurrence based algorithm such as [word2vec](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality) or [GloVe](http://nlp.stanford.edu/projects/glove/)
In the paper, we use GloVe for its speed, but either algorithm should be fine.
Here we release codes to pretrain the code embeddings with GloVe.

1. Use "create_glove_comap.py" with ".seqs" file, which is generated by build_trees.py. (Note that you must run build_trees.py first before training the code embedding)
The execution command is `python create_glove_comap.py <seqs file> <tree file prefix> <output path>`.
This will create a file that contains the co-occurrence information of codes and ancestors.

2. Use "glove.py" on the co-occurrence file generated by create_glove_comap.py.
The execution command is `python glovepy <co-occurrence file> <tree file prefix> <output path>`.
The embedding dimension is set to 128. If you change this, be careful to use the same value when training GRAM.

3. Use the pretrained embeddings when you train GRAM.
The command is `python gram.py <seqs file> <3digitICD9.seqs file> <tree file prefix> <output path> --embed_file <embedding path> --embed_size <embedding dimension>`.
As mentioned above, be sure to set the correct embedding dimension.

**STEP 4: How to prepare your own dataset**

1. GRAM's training dataset needs to be a Python Pickled list of list of list. Each list corresponds to patients, visits, and medical codes (e.g. diagnosis codes, medication codes, procedure codes, etc.)
First, medical codes need to be converted to an integer. Then a single visit can be seen as a list of integers. Then a patient can be seen as a list of visits.
For example, [5,8,15] means the patient was assigned with code 5, 8, and 15 at a certain visit.
If a patient made two visits [1,2,3] and [4,5,6,7], it can be converted to a list of list [[1,2,3], [4,5,6,7]].
Multiple patients can be represented as [[[1,2,3], [4,5,6,7]], [[2,4], [8,3,1], [3]]], which means there are two patients where the first patient made two visits and the second patient made three visits.
This list of list of list needs to be pickled using cPickle. We will refer to this file as the "visit file".

2. The label dataset (let us call this "label file") needs to have the same format as the "visit file".
The important thing is, time steps of both "label file" and "visit file" need to match. DO NOT train GRAM with labels that is one time step ahead of the visits. It is tempting since GRAM predicts the labels of the next visit. But it is internally taken care of.
You can use the "visit file" as the "label file" if you want GRAM to predict the exact codes. 
Or you can use a grouped codes as the "label file" if you are okay with reasonable predictions and want to save time. 
For example, ICD9 diagnosis codes can be grouped into 283 categories by using [CCS](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp) groupers. 
We STRONGLY recommend that you do this, because the number of medical codes can be as high as tens of thousands, 
which can cause not only low predictive performance but also memory issues. (The high-end GPUs typically have only 12GB of VRAM)

3. Use the "build_trees.py" to create ancestor information, using the "visit file". You will also need a mapping file between the actual medical code names (e.g. "419.10") and the integer codes. Please refer to Step 2 to learn how to use "build_trees.py" script.

**STEP 5: Hyper-parameter tuning used in the paper**

This [document](http://www.cc.gatech.edu/~echoi48/docs/gram_hyperparamters.pdf) provides the details regarding how we conducted the hyper-parameter tuning for all models used in the paper.
