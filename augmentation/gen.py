import torch, pdb
import numpy as np
from sample import sample_sequence
import re
import csv
import json
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

EOS_token = '<|endoftext|>'

class GPT2Generator:

    def __init__(self, path, cuda):
       
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
        model_config = GPT2Config(n_embd=768, n_layer=12, n_head=12)       
        self.model = GPT2LMHeadModel(model_config)
        weights = torch.load(path)
        ori_weights=[]
        for key in weights:
            ori_weights.append(key)
        for key in ori_weights:
            if 'module.' in key:
                new_key=re.sub(r'module.', '', key)
                weights[new_key]=weights[key]
                weights.pop(key,None)

        if "lm_head.decoder.weight" in weights:
            weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
            weights.pop("lm_head.decoder.weight",None)

        self.model.load_state_dict(weights)
        self.ix_EOS = 50256
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()


    def tokenize(self, cxt):
        turns = cxt.split(EOS_token)
        ids = []
        for turn in turns:
            ids += self.tokenizer.encode(turn.strip()) + [self.ix_EOS]
        ids = torch.tensor([ids]).view(1, -1)
        if self.cuda:
            ids = ids.cuda()
        return ids
    

    def predict_beam(self, cxt, topk=3, topp=0.8, beam=10, max_t=30): 

        tokens = self.tokenize(cxt)
        len_cxt = tokens.shape[1]
        sum_logP = [0]
        finished = []

        for _ in range(max_t):
            outputs = self.model(tokens)
            predictions = outputs[0]
            logP = torch.log_softmax(predictions[:, -1, :], dim=-1)
            next_logP, next_token = torch.topk(logP, topk)
            sumlogP_ij = []
            sum_prob = 0
            for i in range(tokens.shape[0]):
                for j in range(topk):
                    sum_prob += np.exp(logP[i, j].item())
                    if sum_prob > topp:
                        break
                    if next_token[i, j] == self.ix_EOS:
                        seq = torch.cat([tokens[i, len_cxt:], next_token[i, j].view(1)], dim=-1)
                        if self.cuda:
                            seq = seq.cpu()
                        seq = seq.detach().numpy().tolist()
                        prob = np.exp((sum_logP[i] + next_logP[i, j].item()) / len(seq))
                        hyp = self.tokenizer.decode(seq[:-1])   
                        finished.append((prob, hyp))
                    else:
                        sumlogP_ij.append((sum_logP[i] + next_logP[i, j].item(), i, j))
            if not sumlogP_ij:
                break
            sumlogP_ij = sorted(sumlogP_ij, reverse=True)[:min(len(sumlogP_ij), beam)]
            new_tokens = []
            new_sum_logP = []
            for _sum_logP, i, j in sumlogP_ij:
                new_tokens.append(torch.cat([tokens[i,:], next_token[i, j].view(1)], dim=-1).view(1, -1))
                new_sum_logP.append(_sum_logP)
            tokens = torch.cat(new_tokens, dim=0)
            sum_logP = new_sum_logP

        return finished

    def predict_sampling(self, cxt, temperature=1, n_hyp=5, max_t=30):

        tokens = self.tokenize(cxt)
        tokens = tokens.repeat(n_hyp, 1)
        len_cxt = tokens.shape[1]
        sum_logP = [0] * n_hyp
        live = [True] * n_hyp
        seqs = [[] for _ in range(n_hyp)]
        np.random.seed(2020)
        for _ in range(max_t):
            outputs = self.model(tokens)
            predictions = outputs[0]
            prob = torch.softmax(predictions[:, -1, :] / temperature, dim=-1)
            if self.cuda:
                prob = prob.cpu()
            prob = prob.detach().numpy()
            vocab = prob.shape[-1]
            next_tokens = []
            for i in range(n_hyp):
                next_token = np.random.choice(vocab, p=prob[i,:])
                next_tokens.append(next_token)
                if not live[i]:
                    continue
                sum_logP[i] += np.log(prob[i, next_token])
                seqs[i].append(next_token)
                if next_token == self.ix_EOS:
                    live[i] = False
                    continue
            next_tokens = torch.LongTensor(next_tokens).view(-1, 1)
            if self.cuda:
                next_tokens = next_tokens.cuda()
            tokens = torch.cat([tokens, next_tokens], dim=-1)

        ret = []
        for i in range(n_hyp):
            if live[i]:    
                continue
            prob = np.exp(sum_logP[i] / (len(seqs[i]) + 1))
            hyp = self.tokenizer.decode(seqs[i][:-1])   
            ret.append((prob, hyp))

        return ret
        
    def generate(self, params,finput,foutput):

        with open(finput,'r') as fr,open(foutput,'w') as f2:
            count=0
            for line in fr:
                line=line.strip().split('\t')
                if len(line)==1:
                    continue
                count+=1
                cxt=line[0]
                ret = self.predict(cxt, **params)
                if len(ret)==0:
                    continue
                ans=sorted(ret, reverse=True)
                cxt2=line[1]
                ret2 = self.predict(cxt2, **params)
                if len(ret2)==0:
                    continue
                ans2=sorted(ret2, reverse=True)
                f2.write(ans[0][1]+'\t'+ans2[0][1]+'\t'+line[2]+'\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--sampling', action='store_true')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--beam', type=int, default=3)
    parser.add_argument('--topp', type=float, default=0.8)
    parser.add_argument('--max_n', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--n_hyp', type=int, default=5)
    parser.add_argument('--length', type=int, default=40)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
        
    args = parser.parse_args()

    cuda = False if args.cpu else torch.cuda.is_available()
    generator = GPT2Generator(args.path, cuda)
    if args.sampling:
        params = {'temperature': args.temperature, 'n_hyp': args.n_hyp}
        generator.predict = generator.predict_sampling
    else:
        params = {'topk': args.topk, 'beam': args.beam, 'topp': args.topp}
        generator.predict = generator.predict_beam

    model = generator
    model.generate(params,args.input,args.output)
