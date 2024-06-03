import random
import string
from filelock import SoftFileLock
from os.path import join as pjoin, abspath, dirname
import os, sys
import re
from glob import glob
import json
import time
from pprint import pprint
import traceback
from tqdm import tqdm
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer, util


model_name = 'msmarco-distilbert-base-v4'

model = SentenceTransformer(model_name, device='cuda')


def work_speech(vid):
    if not os.path.exists(f"get_ssg_dataset/sample_raw_dataset/{vid}/paper/paper_sentences.json"):
        if vid.startswith('CHI') or vid.startswith('Ubi'):
            raise ValueError('There are no papers!')
        else:
            return

    with open(f'get_ssg_dataset/sample_raw_dataset/{vid}/paper/paper_sentences.json', 'r') as f:
        sentence_list = json.load(f)
    with open(f'get_ssg_dataset/sample_raw_dataset/{vid}/ocr/seg_speech_ocr.json', 'r', encoding='utf-8') as f:
        seg_speech_ocr = json.load(f)

    querys_embdding = model.encode([seg['speech_text'] for seg in seg_speech_ocr])
    passages_embedding = model.encode(sentence_list)

    if querys_embdding.shape[0] != 0 and passages_embedding.shape[0] != 0:
        cos_similarity = util.cos_sim(querys_embdding, passages_embedding)
    else:
        cos_similarity = None
    
    related_result = []
    for seg_index, seg in enumerate(tqdm(seg_speech_ocr, desc=vid)):
        if cos_similarity is not None:
            paper_sens_indexs = sorted(
                list(range(len(sentence_list))),
                key=lambda i: cos_similarity[seg_index, i],
                reverse=True
            )
            related_sentences = [
                {
                    'sentence': sentence_list[paper_sens_indexs[i]],
                    'cos_similarity': float(cos_similarity[seg_index, paper_sens_indexs[i]])
                }
                for i in range(10)
            ]
        else:
            related_sentences = []
        related_result.append({
            'speech_text': seg['speech_text'],
            'related_sentences': related_sentences
        })
    with open(f'get_ssg_dataset/sample_raw_dataset/{vid}/paper/related_sentences.json', 'w', encoding='utf-8') as f:
        json.dump(related_result, f, indent=2, ensure_ascii=False)
    

for vid in os.listdir("get_ssg_dataset/sample_raw_dataset"):
    work_speech(vid)

