"""
1. Batch size shoud be 1 to reproduce the evaluation result of each sample.
"""
from glob import glob
import json
from pprint import pprint
import string
import sys
import numpy as np

from tqdm import tqdm
from bart_score import BARTScorer
from bert_score import BERTScorer
from transformers import set_seed


def compute_scores(compare_list):
    bert_scorer = BERTScorer(lang="en", batch_size=1, rescale_with_baseline=True, model_type='microsoft/deberta-large-mnli')

    bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='score/bart_score.pth') # From https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing

    preds = [ pred for gold, pred in compare_list]
    golds = [ gold for gold, pred in compare_list]

    set_seed(0)
    bert_score = bert_scorer.score(preds, golds)[2].mean()
    bert_score = f"{float(bert_score):.3f}"
    print(bert_score)

    del bert_scorer

    set_seed(0)
    bart_R = np.array(bart_scorer.score(
        srcs=preds,
        tgts=golds,
        batch_size=1
    ))
    print(bart_R)
    bart_P = np.array(bart_scorer.score(
        srcs=golds,
        tgts=preds,
        batch_size=1
    ))
    print(bart_P)
    bart_score = np.mean(2 * bart_P * bart_R / (bart_P + bart_R))
    bart_score =  f"{float(bart_score):.3f}"
    return {'bert': bert_score, 'bart': bart_score}

def work(in_path):
    in_content = json.load(open(in_path))

    generated_key = [i for i in list(set([ii for i in in_content for ii in i])) if i.startswith('generated')]
    assert len(generated_key) == 1, generated_key + ' ' + in_path
    generated_key = generated_key[0]
    
    for i in in_content:
        assert generated_key[10:] in i, i
        assert generated_key in i, i

    for i in in_content:
        if i[generated_key] == '':
            i[generated_key] = 'NULL'

    result = compute_scores([[i[generated_key[10:]], i[generated_key]] for i in in_content])
   
    with open(in_path.replace('.json', '.eval.brt'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    in_paths = sys.argv[1:]
    if not in_paths:
        raise ValueError('Input is empty!')
    [work(in_path) for in_path in tqdm(in_paths)]

