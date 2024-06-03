from collections import Counter
from itertools import chain
import json
import os
from os.path import join as pjoin
from pprint import pprint
import string
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tokenizer import Tokenizer
from copy import deepcopy as c


set_split = json.load(open('ssg_dataset/sample_set_split.json', encoding='utf-8'))

tokenizer = Tokenizer(model_path='ssg_llama2/statics/debug.llama2_7b/tokenizer.model')

def get_dataset(data_split, args, raw_text=False):
    vid_list = sorted([i for i, j in set_split.items() if j == data_split])
    if args.paper or args.subset:
        vid_list = [i for i in vid_list if i.startswith('CHI') or i.startswith('Ubi')]

    if not args.paper:
        p2l_question_prompt = ["# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a slideshow creator and generate the text in the corresponding single slide based on the given speech text.\n# Speech text:\n", "\n# Slide text:\n"]
        l2p_question_prompt = ["# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text in the given single slide.\n# Slide text:\n", "\n# Speech text:\n"]
    else:
        p2l_question_prompt = ["# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a slideshow creator and generate the text in the corresponding single slide based on the given speech text and related sentences in the paper.\n# Speech text:\n", "\n# Related sentences in the paper:\n", "\n# Slide text:\n"]
        l2p_question_prompt = ["# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text in the given single slide and related sentences in the paper.\n# Slide text:\n", "\n# Related sentences in the paper:\n", "\n# Speech text:\n"]

    dialog_list = []

    for vid in tqdm(vid_list, desc='process data', mininterval=10):
        with open(f'ssg_dataset/text_data/{data_split}/{vid}.json', 'r', encoding='utf-8') as f:
            seg_speech_ocr = json.load(f)

        for unit_index, unit in enumerate(seg_speech_ocr):
            if raw_text:
                process_fn = lambda sentence: c(sentence)
            else:
                process_fn = lambda sentence: tokenizer.encode(sentence, bos=False, eos=False)
            speech_tokens = process_fn(unit['speech_text'])
            ocr_tokens = process_fn(unit['ocr_text'])
            if args.paper:
                paper_tokens = process_fn(unit['paper_text'])

            if args.task == 'p2l':
                question = process_fn(p2l_question_prompt[0]) + speech_tokens + process_fn(p2l_question_prompt[1])
                answer = ocr_tokens
                if args.paper:
                    question += paper_tokens + process_fn(p2l_question_prompt[2])
            elif args.task == 'l2p':
                question = process_fn(l2p_question_prompt[0]) + ocr_tokens + process_fn(l2p_question_prompt[1])
                answer = speech_tokens
                if args.paper:
                    question += paper_tokens + process_fn(l2p_question_prompt[2])
            else:
                raise NotImplementedError
            dialog_list.append({
                'question': question,
                'answer': answer
            })

    return dialog_list


def build_input_from_segments(instance, args):
    """ Build a sequence of input """
    instance['question'].insert(0, tokenizer.bos_id)
    instance['answer'].append(tokenizer.eos_id)

    instance["input_ids"] = list(chain(instance['question'], instance['answer']))
    
    instance["lm_labels"] = ([-100] * len(instance['question'])) + instance['answer']
    
    return instance

class AVSDDataSet(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        instance = self.data[index]
        instance = build_input_from_segments(
            instance,
            self.args)
        input_ids = torch.as_tensor(instance["input_ids"]).long()
        lm_labels = torch.as_tensor(instance["lm_labels"]).long()
            
        return input_ids, lm_labels


def collate_batch(batch, args):

    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float() * pad_token
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result

    input_ids_list, lm_labels_list = [], []

    for i in batch:
        input_ids_list.append(i[0])
        lm_labels_list.append(i[1])

    input_ids = padding(input_ids_list, 0)
    lm_labels_list = padding(lm_labels_list, -100)
    input_mask = input_ids != 0

    return input_ids, lm_labels_list, input_mask


if __name__ == '__main__':
    from args import get_args
    two_args = get_args('train')
    args = two_args[1]
    print(args)

    train_dataset = AVSDDataSet(
        data=get_dataset('dev', args),
        args=args)
    # print(Counter([i[0].size(0) for i in train_dataset]))
    # exit(0)
    train_loader = DataLoader(train_dataset,
                              batch_size=3,
                              shuffle=True,
                              collate_fn=lambda x: collate_batch(x, args))
    for batch in tqdm(train_loader):
        # print(batch)
        print([tokenizer.decode(i.tolist()) for i in batch[0]])

        print(batch[1])

        print(batch[2])
        
        break
    
    
    
    
    
    
    