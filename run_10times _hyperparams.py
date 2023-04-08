import numpy as np
import gram_hf
#import gram_hf_no_attention as gram_hf

if __name__ == '__main__':
    for train_ratio in np.arange(0.07,0.77,0.07):
        print(train_ratio)
        seq_file = 'output2/output2.seqs'
        tree_file = 'output2/output2'
        label_file = 'output/output.hfs'
        out_file = 'gram_hf/gram_hf'
        #out_file = 'gram_hf_no_attention/gram_hf' # for no attention
        inputDimSize = gram_hf.calculate_dimSize(seq_file)
        numClass = 1 # predicte a binary value (heart failure)
        numAncestors = gram_hf.get_rootCode(tree_file+'.level2.pk') - inputDimSize + 1


        # Added by Allan
        # GRAM Hyperparameters
        # Below we try the recommended hyperparameter settings shown in the table
        
        # Test 1: GRAM: Table 5 Row 1: Disease progression modeling 
        H_embDimSize        = 500      # 'm'
        H_hiddenDimSize     = 500      # 'r'
        H_attentionDimSize  = 100      # 'l'
        H_L2                = 0.0001      # 'L2'
        H_dropoutRate       = 0.6      # 'Dropout rate'

        # Test 2: GRAM: Table 5 Row 1: Disease progression modeling (MIMIC-III)
        #H_embDimSize        = 400      # 'm'
        #H_hiddenDimSize     = 400      # 'r'
        #H_attentionDimSize  = 100      # 'l'
        #H_L2                = 0.001      # 'L2'
        #H_dropoutRate       = 0.6      # 'Dropout rate'

        # Test 2: GRAM: Table 5 Row 3: HF prediction (Sutter HF cohort)
        #H_embDimSize        = 200      # 'm'
        #H_hiddenDimSize     = 200      # 'r'
        #H_attentionDimSize  = 100      # 'l'
        #H_L2                = 0.001      # 'L2'
        #H_dropoutRate       = 0.6      # 'Dropout rate'

        # With hyperparameter settings:
        gram_hf.train_GRAM(
            seqFile=seq_file,
            inputDimSize=inputDimSize,
            treeFile=tree_file,
            numAncestors=numAncestors,
            labelFile=label_file,
            numClass=numClass,
            outFile=out_file,
            embFile='',
            embDimSize=         H_embDimSize,        # was 128,
            hiddenDimSize=      H_hiddenDimSize,     # was 128,
            attentionDimSize=   H_attentionDimSize,  # was 128,
            batchSize=100,
            max_epochs=100,#args.n_epochs,
            L2=                 H_L2,           # was 0.001,
            dropoutRate=        H_dropoutRate,  # was 0.5,
            logEps=1e-8,
            train_ratio=train_ratio
        )

        # Original:
        #gram_hf.train_GRAM(
        #    seqFile=seq_file,
        #    inputDimSize=inputDimSize,
        #    treeFile=tree_file,
        #    numAncestors=numAncestors,
        #    labelFile=label_file,
        #    numClass=numClass,
        #    outFile=out_file,
        #    embFile='',
        #    embDimSize=128,
        #    hiddenDimSize=128,
        #    attentionDimSize=128,
        #    batchSize=100,
        #    max_epochs=100,#args.n_epochs,
        #    L2=0.001,
        #    dropoutRate=0.5,
        #    logEps=1e-8,
        #    train_ratio=train_ratio
        #)

        """
        train_GRAM(
            seqFile=args.seq_file,
            inputDimSize=inputDimSize,
            treeFile=args.tree_file,
            numAncestors=numAncestors,
            labelFile=args.label_file,
            numClass=numClass,
            outFile=args.out_file,
            embFile=args.embed_file,
            embDimSize="dimensionality m of the basic embedding e",
            hiddenDimSize="dimensionality r of the RNN hidden layer ht from Eq. (4)",
            attentionDimSize="dimensionality l of Wa and ba from Eq. (3):",
                ^ for the above: 'parser.add_argument('--attention_size', type=int, default=128, help='The dimension size of hidden layer of the MLP that generates the attention weights (default value: 128)')'
            
            batchSize=args.batch_size,
            max_epochs=1,#args.n_epochs,
            L2="L2 regularization coefficient for all weights except RNN weights",
            dropoutRate="dropout rate for the dropout on the RNN hidden layer:",
            logEps=args.log_eps,
            verbose=args.verbose,
            train_ratio= args.train_ratio
        )
        """