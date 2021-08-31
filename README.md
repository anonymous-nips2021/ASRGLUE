# ASR-GLUE

ASR-GLUE is constructed on the basis of GLUE, a popular NLU evaluation benchmark consists of diverse NLU tasks.

We select 5 typical NLU tasks from it, namely:
Sentiment classification (SST-2), Semantic Textual Similarity (STS-B), paraphrase  (QQP QNLI), Recognizing Textual Entailment (RTE) and incorporate with a
Science NLI task (SciTail), resulting in 6 tasks in total.

----------
##### Download ASR-GLUE
You can download ASR-GLUE via google onedrive [Link](https://drive.google.com/drive/folders/1slqI6pUiab470vCxQBZemQZN-a_ssv1Q?usp=sharing)

----

##### ASR-GLUE folder structure
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
##### Data Statisitics 
| Corpus          | Train (text only) |  Dev (audio + text) | Test (audio + text)| 
| :-------------: | :---------------: | :---------------: | :---------------: |
|SST-2                   | 67349                  |2772               | 2790 |
|STS-B                   |5749                    | 3042              | 3222 |                   
QQP                     |363846                   |1476               |3996  |               
RTE                     |2490                     |2070               |2088  |                   
SciTail                 |23596                    |2718               |2736  | 
----