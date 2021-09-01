from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import torch
from sklearn.metrics import matthews_corrcoef, f1_score
log_format = '%(asctime)s %(message)s'
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _read_txt(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                line=line.strip().split('\t')
                lines.append(line)
            return lines
    
class qqpProcessor(DataProcessor):

    def get_train_examples(self, train_file):
        return self._create_examples(
            self._read_txt(train_file), "train")
  
    def get_dev_examples(self, dev_file):
        """See base class."""
        return self._create_examples(
            self._read_txt(dev_file), "dev")

    def get_test_examples(self, test_file):
        """See base class."""
        return self._create_examples(
            self._read_txt(test_file),'test')
    
    def get_labels(self):
        """See base class."""   
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
           
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class qnliProcessor(DataProcessor):

    def get_train_examples(self, train_file):
        return self._create_examples(
            self._read_txt(train_file), "train")
  
    def get_dev_examples(self, dev_file):
        """See base class."""
        return self._create_examples(
            self._read_txt(dev_file), "dev")

    def get_test_examples(self,test_file):
        """See base class."""
        return self._create_examples(
            self._read_txt(test_file),'test')
    
    def get_labels(self):
        """See base class."""   
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class rteProcessor(DataProcessor):

    def get_train_examples(self, train_file):
        return self._create_examples(
            self._read_txt(train_file), "train")
  
    def get_dev_examples(self,dev_file):
        """See base class."""
        return self._create_examples(
            self._read_txt(dev_file), "dev")

    def get_test_examples(self,test_file):
        """See base class."""
        return self._create_examples(
            self._read_txt(test_file),'test')
    
    def get_labels(self):
        """See base class."""   
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            if label=='label':
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class sstProcessor(DataProcessor):  

    def get_train_examples(self, train_file):
        """See base class."""
        return self._create_examples(self._read_txt(train_file), "train")
  
    def get_dev_examples(self,dev_file):
        """See base class."""
        return self._create_examples(self._read_txt(dev_file), "dev")

    def get_test_examples(self, test_file):
        """See base class."""
        return self._create_examples(self._read_txt(test_file),'test')
    
    def get_labels(self):
        """See base class."""   
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class scitailProcessor(DataProcessor):

    def get_train_examples(self, train_file):
        return self._create_examples(self._read_txt(train_file), "train")
  
    def get_dev_examples(self, dev_file):
        """See base class."""
        return self._create_examples(self._read_txt(dev_file), "dev")

    def get_test_examples(self, test_file):
        """See base class."""
        return self._create_examples(self._read_txt(test_file),'test')
    
    def get_labels(self):
        """See base class."""   
        return ["entails", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b =line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class stsbProcessor(DataProcessor):
    def get_train_examples(self, train_file):
        """See base class."""
        return self._create_examples(self._read_txt(train_file), "train")

    def get_dev_examples(self, dev_file):
        """See base class."""
        return self._create_examples(self._read_txt(dev_file), "dev")

    def get_test_examples(self,test_file):
        """See base class."""
        return self._create_examples(self._read_txt(test_file),'test')


    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a=line[0]
            text_b=line[1]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    if output_mode == "classification":
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
        
    if task_name=="sts-b":
        return pearson_and_spearman(preds,labels)
    else:
        return acc_and_f1(preds,labels)


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(task_name,result, file_name,train_file,test_file):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % ("train_file: ",train_file))
        writer.write("%s = %s\n" % ("test_file: ", test_file))
        if task_name=="sts-b":
            writer.write("%s = %s\n" % ("spearmanr", str(result["spearmanr"])))
        else:
            writer.write("%s = %s\n" % ("acc", str(result["acc"])))
        writer.write("\n")


def do_eval(model, task_name, eval_dataloader,device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
        # if output_mode == "classification":
        #     with torch.no_grad():
        #         # outputs = model(input_ids,labels=label_ids)
        #         # loss,logits=outputs[:2]
        #         tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)   
        #     # loss_fct = CrossEntropyLoss()
        #     # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        # elif output_mode == "regression":
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    
    parser.add_argument("--task_name",default=None,type=str,required=True,help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
   
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    
    parser.add_argument("--do_test",action='store_true',help="")
    parser.add_argument('--seed',type=int,default=42,help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--eval_step',type=int,default=50)

    parser.add_argument("--train_file",type=str,help="train file")  
    parser.add_argument("--dev_file",type=str,help="dev file")
    parser.add_argument("--test_file",type=str,help="test file")

    parser.add_argument('--temperature',type=float,default=1.)

    args = parser.parse_args()
    processors = {
        "sst-2": sstProcessor,
        "qqp": qqpProcessor,
        'scitail':scitailProcessor,
        'qnli':qnliProcessor,
        'rte':rteProcessor, 
        'sts-b':stsbProcessor
    }
    output_modes = {
        "qqp": "classification",
        "sst-2": "classification",
        "scitail": "classification",
        "qnli": "classification",
        'rte':"classification",
        'sts-b':"regression"
    }
    num_labels_task = {
        "sst-2": 2,
        "qqp": 2,
        'scitail':2,
        'qnli':2,
        'rte':2,
        'sts-b':1
    }
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
              num_labels =num_labels_task[args.task_name])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.do_test:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    test_examples = processor.get_test_examples(args.test_file)
    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    test_data, test_labels = get_tensor_data(output_mode, test_features)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    output_test_file='test_result.txt'

    if args.do_test:
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
        model.to(device)
        model.eval()
 
        test_result = do_eval(model, task_name,test_dataloader,device, output_mode, test_labels, num_labels)
        result_to_file(task_name,test_result,output_test_file,args.train_file,args.test_file)

        return 0

    train_examples = processor.get_train_examples(args.train_file)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
    train_data, _ = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_examples = processor.get_dev_examples(args.dev_file)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.to(device)
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    loss_mse = MSELoss()
    global_step = 0
    best_dev,best_test= 0.0,0.0
    save=0

    for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0.

        model.train()
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            if input_ids.size()[0] != args.train_batch_size:
                    continue
            if output_mode=="classification":
                #outputs = model(input_ids, labels=label_ids)
                #loss, logits = outputs[:2]    
                loss = model(input_ids, segment_ids, input_mask, label_ids)   
            else:
                logits = model(input_ids, segment_ids, input_mask)     
                loss = loss_mse(logits.view(-1), label_ids.view(-1))
            if n_gpu > 1:
                loss = loss.mean()  
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
           
            nb_tr_examples += label_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (global_step + 1) % args.eval_step== 0:
                model.eval()
                loss = tr_loss / (step + 1)
                result = {}
                result = do_eval(model, task_name, eval_dataloader,device, output_mode, eval_labels, num_labels)
                result['global_step'] = global_step
                result['loss'] = loss
                logger.info("***** dev results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                if output_mode=="classification":
                    if result['acc']>best_dev:
                        best_dev=result['acc']
                        save=1
                    else:
                        save=0
                if output_mode=="regression" :
                    if result['spearmanr']>best_dev:
                        best_dev=result["spearmanr"]
                        save=1
                    else:
                        save=0
                if save:
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    # output_config_file=os.path.join(args.output_dir,'config.json')
                    torch.save(model_to_save.state_dict(), output_model_file)
                    # model_to_save.config.to_json_file(output_config_file)
                    # tokenizer.save_vocabulary(args.output_dir)
                    test_result = do_eval(model, task_name,test_dataloader,device, output_mode, test_labels, num_labels)
                    logger.info("***** Test results *****")
                    for key in sorted(test_result.keys()):
                        logger.info("  %s = %s", key, str(test_result[key]))
                model.train()


if __name__ == "__main__":
    main()