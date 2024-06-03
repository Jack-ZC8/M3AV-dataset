import argparse
from collections import Counter
from itertools import chain
import json
import os
from os.path import join as pjoin
from pprint import pprint
import shutil
import string
from tqdm import tqdm
import random
from copy import deepcopy as c



set_split = json.load(open('get_ssg_dataset/set_split.json', encoding='utf-8'))

SAVE_DIR = 'ssg_dataset'

def save_vid(vid, data_split):
    with open(f'get_ssg_dataset/sample_raw_dataset/{vid}/ocr/seg_speech_ocr.json', 'r', encoding='utf-8') as f:
        seg_speech_ocr = json.load(f)

    if vid[:3] in ['CHI', 'Ubi']:
        with open(f'get_ssg_dataset/sample_raw_dataset/{vid}/paper/related_sentences.json', 'r', encoding='utf-8') as f:
            related_json = json.load(f)
        assert len(related_json) == len(seg_speech_ocr)

    cut_words_func = lambda sent, max_words: ' '.join(sent.split(' ')[:max_words])

    vid_res = []
    for seg_index, seg in enumerate(seg_speech_ocr):
        speech_text = cut_words_func(seg['speech_text'], args.max_lp_words)
        ocr_text = cut_words_func(seg['ocr_text'], args.max_lp_words)

        if vid[:3] in ['CHI', 'Ubi']:
            related_sent_ls = [i['sentence'] for i in related_json[seg_index]['related_sentences'][:args.related_tops]
                                        if i['cos_similarity'] > args.related_thre]
            paper_text = '\n'.join([
                    cut_words_func(i, args.max_papersent_words)
                    for i in related_sent_ls
                ])
            vid_res.append({
                'speech_text': speech_text,
                'ocr_text': ocr_text,
                'paper_text': paper_text,
                'image_name': seg['name']
            })
        else:
            vid_res.append({
                'speech_text': speech_text,
                'ocr_text': ocr_text,
                'image_name': seg['name']
            })
    with open(pjoin(SAVE_DIR, 'text_data', data_split, vid+'.json'), 'w', encoding='utf-8') as f:
        json.dump(vid_res, f, indent=2, ensure_ascii=False)
    
    os.makedirs(pjoin(SAVE_DIR, 'image_data'), exist_ok=True)
    os.popen(f'cp get_ssg_dataset/sample_raw_dataset/{vid}/ocr/seg_imgs/*.jpg {SAVE_DIR}/image_data/').read()
        

def save_dataset(args):
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    with open(pjoin(SAVE_DIR, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, indent=2, ensure_ascii=False)
    
    for data_split in ['train', 'dev', 'test']:
        vid_list = [i for i in os.listdir("get_ssg_dataset/sample_raw_dataset") if set_split[i] == data_split]
        os.makedirs(pjoin(SAVE_DIR, 'text_data', data_split))
        for vid in tqdm(vid_list):
            save_vid(vid, data_split)
    with open("ssg_dataset/sample_set_split.json", "w", encoding='utf-8') as f:
        json.dump({i: j for i, j in set_split.items() if i in os.listdir("get_ssg_dataset/sample_raw_dataset")}, indent=2, fp=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ======================= data ======================= #
    parser.add_argument("--max_lp_words", type=int, default=400)
    parser.add_argument("--max_papersent_words", type=int, default=100)
    parser.add_argument("--related_thre", type=float, default=0.5)
    parser.add_argument("--related_tops", type=int, default=3)

    args = parser.parse_args()

    save_dataset(args)
    
 