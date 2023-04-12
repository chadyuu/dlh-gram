import numpy as np
import gram_hf

seq_file = 'output2/output2.seqs'
tree_file = 'output2/output2'
label_file = 'output/output.hfs'
inputDimSize = gram_hf.calculate_dimSize(seq_file)
numClass = 1 # predicte a binary value (heart failure)
numAncestors = gram_hf.get_rootCode(tree_file+'.level2.pk') - inputDimSize + 1

def train_gram_hyperparam(embDimSize=200, hiddenDimSize=100,attentionDimSize=100,L2=0.001,dropoutRate=0.6, out_file=''):
    gram_hf.train_GRAM(
        seqFile=seq_file,
        inputDimSize=inputDimSize,
        treeFile=tree_file,
        numAncestors=numAncestors,
        labelFile=label_file,
        numClass=numClass,
        outFile=out_file,
        embFile='',
        embDimSize=embDimSize, 
        hiddenDimSize=hiddenDimSize,
        attentionDimSize=attentionDimSize,
        batchSize=100,
        max_epochs=10,
        L2=L2, 
        dropoutRate=dropoutRate,
        logEps=1e-8,
        train_ratio=0.7
    )

if __name__ == '__main__':
    # default settting
    train_gram_hyperparam(out_file='hyperparam/default')

    # change parameters
    for embDimSize in [100,300,400,500]:
        train_gram_hyperparam(embDimSize=embDimSize, out_file='hyperparam/embDimSize'+str(embDimSize))
    for hiddenDimSize in [200,300,400,500]:
        train_gram_hyperparam(hiddenDimSize=hiddenDimSize, out_file='hyperparam/hiddenDimSize'+str(hiddenDimSize))
    for attentionDimSize in [200,300,400,500]:
        train_gram_hyperparam(attentionDimSize=attentionDimSize, out_file='hyperparam/attentionDimSize'+str(attentionDimSize))
    for L2 in [0.0001,0.01, 0.1]:
        train_gram_hyperparam(L2=L2, out_file='hyperparam/L2'+str(L2))
    for dropoutRate in [0.0,0.2,0.4,0.8]:
        train_gram_hyperparam(dropoutRate=dropoutRate, out_file='hyperparam/dropoutRate'+str(dropoutRate))

    train_gram_hyperparam(embDimSize=500, hiddenDimSize=500, out_file='hyperparam/embDimSize500_hiddenDimSize500')

    # possibly best
    train_gram_hyperparam(
        embDimSize=400, 
        hiddenDimSize=400, 
        attentionDimSize=200,
        L2=0.0001,
        dropoutRate=0.8,
        out_file='hyperparam/best')
