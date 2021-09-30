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
|  |  ├── N-best list ASR results
|  |  |  ├── hyp.*

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
Detailed WER results in ASRGLUE_WER.xlsx 

### Noise audito data simulation
Please first download the room impulse response and backgroud noise data via:
[Link](http://www.openslr.org/resources/28/rirs\_noises.zip)
Then you can use the scripts to simulated the nosie:
```
bash scripts/make_rvb_nosie.sh --SNR $i --norvb_datadir=YourCleanspeechDir 
```
You need to install [kaldi](https://github.com/kaldi-asr/kaldi) to use this script
Note that YourCleanspeechDir is a Dir in kaldi format.
Which needs to contain four files:
```
wav.scp  # save the path of your clean wav files: ID PATH 
utt2spk 
spk2utt 
text     # The transcripts: ID Text
```
We provide an example in examples/ssb_2

If you want to simulate with signal samlping noise, please 
```
tar -xvf colorednoise.tar
cp -r colorednoise RIRS_NOISES
```
----
### ASR server
Currently you can use google Speech-to-text API to reproduce our google ASR results:
Note that your need to set up the google API account first to use the script [link](https://cloud.google.com/speech-to-text/?utm_source=google&utm_medium=cpc&utm_campaign=japac-AU-all-en-dr-bkws-all-super-trial-e-dr-1009882&utm_content=ims_text-ad-none-none-DEV_c-CRE_507092309996-ADGP_Hybrid%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20~%20AI%20%26%20ML%20~%20Speech-to-Text_Travel%20-%20Speech%20-%20google%20speech%20to%20text-KWID_43700060575389626-kwd-21425535976&userloc_9061630-network_g&utm_term=KW_google%20speech%20to%20text&gclid=CjwKCAjwndCKBhAkEiwAgSDKQdUXC365lVSK1qwk1MbSMrisNXmqCge269p1XsphN41u-GSuuIhtqRoCEr0QAvD_BwE&gclsrc=aw.ds)
```
python scripts/recg_google.py wav.scp
```

We will provided our kaldi based online server  at the end of October 2021


----
### ASR-Robust NLU

1. If you want to test the robustness of your NLU model, please replace the $BERT_BASE with your NLU model in test.sh, then run:

    $ sh scripts/test.sh

2. If you want to train a new NLU model, please run:

    $ sh scripts/train.sh

3. For correction-based methods:
   
   GECToR:  sh scripts/gec.sh 

   BART-C:  sh scripts/bart-c.sh 
   
   then modify $TEST_FILE in test.sh with the output of your correction model.
   
4. For augmentation-based methods:

    CM:  sh scripts/cm.sh 
    
    GPT-2: sh scripts/gpt2.sh 
    
    BART-S: sh scripts/bart-s.sh 
    
   then modify $TRAIN_FILE in train.sh.
----





### License
This work is licensed under a Creative Commons “Attribution 4.0 International” license
