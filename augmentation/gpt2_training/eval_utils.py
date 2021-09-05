import torch
import logging

import numpy as np

from mt.bleu.bleu import Bleu
from collections import defaultdict

logger = logging.getLogger(__name__)

EOS_ID = 50256
SEP= '<SEP>'
EOS = '<EOS>'

def cal_BLEU_4(generated, reference, is_corpus=False):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]},
                                                  {0: [g]})
        for i, s in zip([0, 1, 2, 3], score):
            BLEUscore[i] += s
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score

def eval_model_loss(model, tokenizer, eval_dataloader,epoch_id, args):
    
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]
            loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)

def tokenize(tokenizer,cxt):

    ids = tokenizer.encode(cxt.strip()) + [EOS_ID]
    ids = torch.tensor([ids]).view(1, -1)
    ids = ids.cuda()

    return ids


def predict_beam(model,toker,cxt, topk=3, topp=0.8, beam=10, max_t=30):

        tokens = tokenize(toker,cxt)
        len_cxt = tokens.shape[1]
        sum_logP = [0]
        finished = []

        for _ in range(max_t):
            outputs = model(tokens)
           
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
                    if next_token[i, j] == EOS_ID:
                        seq = torch.cat([tokens[i, len_cxt:], next_token[i, j].view(1)], dim=-1)
                
                        seq = seq.cpu()
                        seq = seq.detach().numpy().tolist()
                        prob = np.exp((sum_logP[i] + next_logP[i, j].item()) / len(seq))
                        hyp = toker.decode(seq[:-1])  
                        finished.append((prob, hyp))
                    else:
                        sumlogP_ij.append((
                            sum_logP[i] + next_logP[i, j].item(), 
                            i, j))
                
            if not sumlogP_ij:
                break
            sumlogP_ij = sorted(sumlogP_ij, reverse=True)[:min(len(sumlogP_ij), beam)]
            new_tokens = []
            new_sum_logP = []
            for _sum_logP, i, j in sumlogP_ij:
                new_tokens.append(
                        torch.cat([tokens[i,:], next_token[i, j].view(1)], dim=-1).view(1, -1)
                        )
                new_sum_logP.append(_sum_logP)
            tokens = torch.cat(new_tokens, dim=0)
            sum_logP = new_sum_logP

        return finished


def predict_sampling(model,toker, cxt, temperature=1, n_hyp=5, max_t=30):
        """ sampling tokens based on predicted probability """

        tokens = tokenize(toker,cxt)
        tokens = tokens.repeat(n_hyp, 1)
        len_cxt = tokens.shape[1]
        sum_logP = [0] * n_hyp
        live = [True] * n_hyp
        seqs = [[] for _ in range(n_hyp)]
        np.random.seed(2020)
        for _ in range(max_t):
            outputs = model(tokens)
            predictions = outputs[0]
            prob = torch.softmax(predictions[:, -1, :] / temperature, dim=-1)
            
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
                if next_token == EOS_ID:
                    live[i] = False
                    continue
            next_tokens = torch.LongTensor(next_tokens).view(-1, 1)
            next_tokens = next_tokens.cuda()
            tokens = torch.cat([tokens, next_tokens], dim=-1)

        ret = []
        for i in range(n_hyp):
            if live[i]:     
                continue
            prob = np.exp(sum_logP[i] / (len(seqs[i]) + 1))
            hyp = toker.decode(seqs[i][:-1])   # strip EOS
            ret.append((prob, hyp))
        return ret
        

        
