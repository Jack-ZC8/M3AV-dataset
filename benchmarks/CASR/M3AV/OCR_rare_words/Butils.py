from __future__ import division

import argparse
import json
import logging
import math
import os
import random
from copy import deepcopy
from time import time
import sys
import editdistance
import numpy as np
import six
import torch

from espnet2.text.build_tokenizer import build_tokenizer
from espnet.lm.lm_utils import make_lexical_tree

random.seed(0)


class BiasProc(object):
    def __init__(self, blist, maxlen, bdrop, bpemodel, charlist):
        self.ocr={}
        dev_ocr_file = open('data/dev/dev_ocr.txt','r')
        test_ocr_file = open('data/test/test_ocr.txt', 'r')
        for line in dev_ocr_file.readlines():
            wav_id, ocr = line.split(' ',1)
            self.ocr[wav_id]=ocr
        for line in test_ocr_file.readlines():
            wav_id, ocr = line.split(' ',1)
            self.ocr[wav_id]=ocr

        with open('local/biasing_list_HCI.txt') as fin:
            self.HCI_wordblist = [line.split() for line in fin]
        with open('local/biasing_list_BIO.txt') as fin:
            self.BIO_wordblist = [line.split() for line in fin]
        with open('local/biasing_list_MATH.txt') as fin:
            self.MATH_wordblist = [line.split() for line in fin]
        with open(blist) as fin:
            self.wordblist = [line.split() for line in fin]
        self.tokenizer = build_tokenizer("bpe", bpemodel)
        self.encodedset = self.encode_blist()
        self.HCI_encodedset, self.BIO_encodedset, self.MATH_encodedset = self.encode_blist_set()
        self.maxlen = maxlen
        self.bdrop = bdrop
        self.chardict = {}
        for i, char in enumerate(charlist):
            self.chardict[char] = i
        self.charlist = charlist


    def encode_blist(self):
        encodedset = set()
        self.encodedlist = []
        for word in self.wordblist:
            bpeword = self.tokenizer.text2tokens(word)
            encodedset.add(tuple(bpeword[0]))
            self.encodedlist.append(tuple(bpeword[0]))
        return encodedset

    def encode_blist_set(self):
        HCI_encodedset = set()
        BIO_encodedset = set()
        MATH_encodedset = set()

        self.HCI_encodedlist = []
        self.BIO_encodedlist = []
        self.MATH_encodedlist = []

        for word in self.HCI_wordblist:
            bpeword = self.tokenizer.text2tokens(word)
            HCI_encodedset.add(tuple(bpeword[0]))
            self.HCI_encodedlist.append(tuple(bpeword[0]))
        for word in self.BIO_wordblist:
            bpeword = self.tokenizer.text2tokens(word)
            BIO_encodedset.add(tuple(bpeword[0]))
            self.BIO_encodedlist.append(tuple(bpeword[0]))
        for word in self.MATH_wordblist:
            bpeword = self.tokenizer.text2tokens(word)
            MATH_encodedset.add(tuple(bpeword[0]))
            self.MATH_encodedlist.append(tuple(bpeword[0]))

        return HCI_encodedset, BIO_encodedset, MATH_encodedset

    def encode_spec_blist(self, blist):
        encoded = []
        for word in blist:
            bpeword = self.tokenizer.text2tokens(word)
            encoded.append(tuple(bpeword))
        return encoded

    def construct_blist(self, bwords):
        if len(bwords) < self.maxlen:
            distractors = random.sample(self.encodedlist, k=self.maxlen - len(bwords))
            sampled_words = []
            for word in distractors:
                if word not in bwords:
                    sampled_words.append(word)
            sampled_words = sampled_words + bwords
        else:
            sampled_words = bwords
        uttKB = sorted(sampled_words)
        worddict = {word: i + 1 for i, word in enumerate(uttKB)}
        lextree = make_lexical_tree(worddict, self.chardict, -1)
        return lextree

    def construct_blist_infer(self, keys, bwords):
        #print(keys[0])
        wav_id = keys[0].split('_')[0]
        ocr = self.encode_spec_blist(self.ocr[wav_id].split())
        sampled_words = []
        for word in ocr:
            if word not in bwords:
                sampled_words.append(word)
        sampled_words = sampled_words + bwords
        '''
        if len(bwords)+len(ocr) < self.maxlen:
            bwords = bwords + ocr
            set_name = wav_id.split('-')[0]
            if len(bwords) < self.maxlen:
                if set_name=='CHI' or set_name=='Ubi':
                    distractors = random.sample(self.HCI_encodedlist, k=self.maxlen - len(bwords))
                elif set_name=='NIH' or set_name=='IPP':
                    distractors = random.sample(self.BIO_encodedlist, k=self.maxlen - len(bwords))
                elif set_name=='MLS':
                    distractors = random.sample(self.MATH_encodedlist, k=self.maxlen - len(bwords))
                else:
                    print(set_name)
                    sys.exit()
                sampled_words = []
                for word in distractors:
                    if word not in bwords:
                        sampled_words.append(word)
                sampled_words = sampled_words + bwords
        else:
            sampled_words = bwords + ocr
        '''
        uttKB = sorted(sampled_words)
        worddict = {word: i + 1 for i, word in enumerate(uttKB)}
        lextree = make_lexical_tree(worddict, self.chardict, -1)
        return lextree

    def select_biasing_words(self, yseqs, suffix=True):
        bwords = []
        wordbuffer = []
        yseqs = [[idx for idx in yseq if idx != -1] for yseq in yseqs]
        for i, yseq in enumerate(yseqs):
            for j, wp in enumerate(yseq):
                wordbuffer.append(self.charlist[wp])
                if suffix and self.charlist[wp].endswith("▁"):
                    if tuple(wordbuffer) in self.encodedset:
                        bwords.append(tuple(wordbuffer))
                    wordbuffer = []
        bwords = [word for word in bwords if random.random() > self.bdrop]
        lextree = self.construct_blist(bwords)
        return bwords, lextree
