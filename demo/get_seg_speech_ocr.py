from collections import Counter
from itertools import chain
import json
import os
from os.path import join as pjoin
from pprint import pprint
import string
from tqdm import tqdm

def process_vid(vid):
    ocr_data = json.load(open(f'{vid}/ocr_result/ocr_result.json', encoding='utf-8'))
    speech_data = json.load(open(f'{vid}/speech/final+timestamps.json', encoding='utf-8'))
    all_words_written = list(chain(*[i['words_written'] for i in speech_data]))

    for image_index, image_data in enumerate(ocr_data):
        
        unique_index_para = set(i['index_para'] for i in image_data['ocr_data'])
        unique_index_para = sorted(list(unique_index_para))
        
        paras_ls = []
        for index in unique_index_para:
            para_text = ' '.join([i['transcription'] for i in image_data['ocr_data'] if i['index_para'] == index])
            # Inspired by https://github.com/IBM/document2slides/blob/main/sciduet-build/extract_slides.py#L33
            nospace = ''.join(para_text.split())
            if len(nospace) == 0:
                continue
            if sum(c in string.ascii_letters for c in nospace) / len(nospace) < 0.5:
                continue
            paras_ls.append(para_text)
        
        sum_para_text = '\n'.join(paras_ls)
        image_data['ocr_text'] = sum_para_text
        # print(sum_para_text)
        # print('='*89)
        del image_data['ocr_data']
        #############################################
        start_s = 0 if image_index == 0 else int(ocr_data[image_index-1]['name'][10:17]) / 1000
        end_s = int(ocr_data[image_index]['name'][10:17]) / 1000
        image_data['speech_text'] = ' '.join(i['word'] for i in all_words_written
                          if start_s <= (i['start'] + i['end']) / 2 < end_s)
        image_data['start'], image_data['end'] = start_s, end_s
    ocr_data = [i for i in ocr_data if i['speech_text'] and i['ocr_text']]
    
    with open(f'{vid}/ocr_result/seg_speech_ocr.json', 'w', encoding='utf-8') as f:
        json.dump(ocr_data, f, indent=2, ensure_ascii=False)

