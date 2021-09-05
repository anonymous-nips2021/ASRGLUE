# ASR-GLUE
---------
### Introduction
ASR-robust General Language Understanding Evaluation (ASR-GLUE) is constructed on the basis of GLUE for the purpose of comprehensively investigate how ASR error affect NLU capability. 

It contains a new collection of 6 different NLU tasks for evaluating the performance of models under ASR error across 3 different levels of background noise and 6 speakers with various voice characteristics.
* **SST-2** The Stanford Sentiment Treebank is a single-input understanding task for sentiment classification. The task is to predict the sentiment of a given sentence in movie reviews domain. Accuracy (ACC) of the binary classification (positive or negative) is used as the metric.
* **STS-B** The Semantic Textual Similarity Benchmark consists of sentence pairs drawn from news headlines, video and image captions, and natural language inference data. The task is to predict sentence similarity scores which ranges from 1 to 5. We evaluate using Pearson and Spearman correlation coefficients.
* **QQP**   The Quora Question Pairs dataset consists of question pairs in social QA questions domain. The task is to determine whether a pair of questions are semantically equivalent. Accuracy (ACC) is used as the metric.
* **QNLI**  Question-answering NLI is modified from the Stanford Question Answering dataset. This is a  sentence pair classification task which determines whether the context sentence contains the answer to the question. Accuracy (ACC) is used as the metric.
* **RTE**   The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual entailment challenges. All datasets are combined and converted to two-class classification: entailment and not entailment. Accuracy (ACC) is used as the metric.
* **Scitail** SciTail is a recently released challenging textual entailment dataset collected from the science domain. This is a natural language inference task which 
determines if a natural language hypothesis can be justifiably inferred from a given premise. Accuracy (ACC) is used as the metric.


----------
### Download ASR-GLUE
The text form of the training set and both the audio (~90hours) and the text of the dev and text set are provided in ASR-GLUE.

You can download ASR-GLUE via google onedrive [Link](https://drive.google.com/drive/folders/1slqI6pUiab470vCxQBZemQZN-a_ssv1Q?usp=sharing)

----

### ASR-GLUE folder structure
```
ASR-GLUE dataset
├── Train set
|  ├── list of NLU tasks    # qnli  qqp  rte  scitail  sst-2  sts-b
|  |  ├── golden            # original training set
|  |  ├──cm                 # text-level augmentation by Confusion Matrix
|  |  ├──gpt                # text-level augmentation by GPT-2
|  |  ├──bart               # text-level augmentation by BART-S
|  |  ├──audio              # audio-level augmentation
├── Test and Dev set
|  ├── list of NLU tasks    # qnli  qqp  rte  scitail  sst-2  sts-b
|  |  ├── Wav folders
|  |  |  ├── Noise level    # low/medium/high
|  |  |  |  ├── Speaker     # Speaker0001 - Speaker0006
|  |  |  |  |  ├──*.wav
|  |  ├── ASR results       # kaldiasr_results and googleasr_results 
|  |  |  ├── hyp.*          # ASR hypothesis cross different speaker and noisy level
|  |  |  ├── groundtruth    # The ground truth transcription
|  |  |  ├── wer            # Summary of word-error-rate (wer) 

```

-----
### Data Statisitics 
| Corpus          | Train (text only) |  Dev (audio + text) | Test (audio + text)| 
| :-------------: | :---------------: | :---------------: | :---------------: |
|SST-2                   | 67349                  |2772               | 2790 |
|STS-B                   |5749                    | 3042              | 3222 |                   
QQP                     |363846                   |1476               |3996  | 
QNLI                    |104743                   |2718             |2718|            
RTE                     |2490                     |2070               |2088  |                   
SciTail                 |23596                    |2718               |2736  | 
----


------
### ASR-Robust NLU

1. If you want to test the robustness of your NLU model, please replace the $BERT_BASE with your NLU model in test.sh, then run:

    $ sh scripts/test.sh

2. If you want to train a new NLU model, please run:

    $ sh scripts/train.sh

3. You can also do ASR correction, e.g., sh scripts/gec.sh then modify $TEST_FILE in test.sh with the output of your correction model.
   
   Moreover, you can also do data augmentation, e.g., sh scripts/gen.sh then modify $TRAIN_FILE in train.sh. Enjoy it!

----





### License
This work is licensed under a Creative Commons “Attribution 4.0 International” license
