import numpy as np
import gram_hf_no_attention as gram_hf

if __name__ == '__main__':
    for train_ratio in np.arange(0.07,0.77,0.07):
        print(train_ratio)
        seq_file = 'output2/output2.seqs'
        tree_file = 'output2/output2'
        label_file = 'output/output.hfs'
        out_file = 'gram_hf_no_attention/gram_hf' # for no attention
        inputDimSize = gram_hf.calculate_dimSize(seq_file)
        numClass = 1 # predicte a binary value (heart failure)
        numAncestors = gram_hf.get_rootCode(tree_file+'.level2.pk') - inputDimSize + 1

        gram_hf.train_GRAM(
            seqFile=seq_file,
            inputDimSize=inputDimSize,
            treeFile=tree_file,
            numAncestors=numAncestors,
            labelFile=label_file,
            numClass=numClass,
            outFile=out_file,
            embFile='',
            embDimSize=128,
            hiddenDimSize=128,
            attentionDimSize=128,
            batchSize=100,
            max_epochs=100,#args.n_epochs,
            L2=0.001,
            dropoutRate=0.5,
            logEps=1e-8,
            train_ratio=train_ratio
        )