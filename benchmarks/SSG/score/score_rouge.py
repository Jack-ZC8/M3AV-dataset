from glob import glob
import json
from pprint import pprint
import string
import sys
from nltk import PorterStemmer
from rouge import Rouge
from spacy.lang.en import English
from tqdm import tqdm



# https://github.com/IBM/document2slides/blob/main/d2s-model/test.py#L61
def compute_rouge(compare_list):
    stemmer = PorterStemmer()
    rouge = Rouge()
    tokenizer = English().tokenizer

    def preprocess(sentence):
        sentence = " ".join([stemmer.stem(str(w))
                        for w in tokenizer(sentence)])
        return sentence

    preds = [ preprocess(pred)
             for gold, pred in compare_list]
    golds = [ preprocess(gold)
             for gold, pred in compare_list]
    scores = rouge.get_scores(preds, golds, avg=True)
    return scores

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

    result = compute_rouge([[i[generated_key[10:]], i[generated_key]] for i in in_content])
    for rouge_type in result:
        for i in ['f']:
            tmp = result[rouge_type][i]
            del result[rouge_type][i]
            result[rouge_type][i] = f"{100 * tmp:.1f}"
    with open(in_path.replace('.json', '.eval.rouge'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    in_paths = sys.argv[1:]
    if not in_paths:
        raise ValueError('Input is empty!')

    [work(in_path) for in_path in tqdm(in_paths)]

