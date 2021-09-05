import random
import string
import numpy as np
import re
import json
import sys
import logging
import argparse

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def _is_valid(string):
    return True if not re.search('[^a-z]', string) else False

def attack_word(word,confusion):

    wlist,plist=[],[]
    for i in confusion[word]:
        wlist.append(i)
        plist.append(confusion[word][i])
    plist=np.asarray(plist)
    plist = plist / np.sum(plist)
    ss=np.random.choice(wlist,p=plist)
    return ss


def random_attack(tok,confusion):
    prob = np.random.random()
    if prob < 0.5: 
        tok_flaw = attack_word(tok,confusion)
        return 1, tok_flaw
    else:
        return 0, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",type=str,default="cm.json")  
    parser.add_argument("--input",type=str,help="input")
    parser.add_argument("--output",type=str,help="output")

    args = parser.parse_args()
    
    with open(args.data_file) as f:
	    confusion=json.load(f)

    with open(args.input,'r') as fr,open(args.output,'w') as fw:
        count=0
        for line in fr:
            line=line.strip().split('\t')
            count+=1
            sen1=line[0].strip().split(' ')
            noisy_line=[]
            for tok in sen1:
                if not _is_valid(tok) or tok not in confusion:
                    noisy_line.append(tok)
                    continue
                _,tok_flaw = random_attack(tok,confusion)
                noisy_line.append(tok_flaw)
            noisy_sen1=' '.join(noisy_line)

            sen2=line[1].strip().split(' ')
            noisy_line=[]
            for tok in sen2:
                if not _is_valid(tok) or tok not in confusion:
                    noisy_line.append(tok)
                    continue
                _,tok_flaw = random_attack(tok)
                noisy_line.append(tok_flaw)
            noisy_sen2=' '.join(noisy_line)
            fw.write(noisy_sen1+'\t'+noisy_sen2+'\t'+line[2]+'\n')

if __name__ == "__main__":
    main()
