# TCPGen with GNN
 - Code available through ESPnet Github PR [link](https://github.com/espnet/espnet/pull/5261)
 - Two versions of biasing list selection method are provided: random distractors and OCR rare words.

### Distractors
1. Download ESPnet PR:
```
git clone https://github.com/espnet/espnet.git
cd espnet
git fetch origin pull/5261/head:pr5261
git checkout pr5261
```

2. Put this recipe in espnet/egs2, and run the training and inferring recipe.
```
cp -r M3AV-dataset/benckmarks/CASR/M3AV espnet/egs2
```
3. The 'all rare words list' is placed in 'asr1_biasing/local', which is mentioned in our paper.

### OCR rare words
1. The OCR rare words list is provided. You can also retrieve in the dataset OCR by yourself.
2. You need to change the inferring process, which means some codes need to be refined. 
```
rm espnet/espnet2/text/Butils.py
cp M3AV-dataset/benckmarks/CASR/M3AV/OCR_rare_words/Butils.py espnet/espnet2/text
rm espnet/espnet2/bin/asr_inference.py
cp M3AV-dataset/benckmarks/CASR/M3AV/OCR_rare_words/asr_inference.py espnet/espnet2/bin
```
3. put the OCR rare words list to data direction.
```
#replace 'espnet2/text/Butils.py'
cp M3AV-dataset/benckmarks/CASR/M3AV/OCR_rare_words/dev_ocr.txt espnet/egs2/M3AV/asr1_biasing/data/dev
cp M3AV-dataset/benckmarks/CASR/M3AV/OCR_rare_words/test_ocr.txt espnet/egs2/M3AV/asr1_biasing/data/test
```
4. Run this recipe.
```
cd espnet/egs2/M3AV/asr1_biasing
./run.sh
```
### Citation
If you find this code useful for your research, please cite the following papers
```
@inproceedings{sun2021tree,
  title={Tree-constrained pointer generator for end-to-end contextual speech recognition},
  author={Sun, Guangzhi and Zhang, Chao and Woodland, Philip C},
  booktitle={2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={780--787},
  year={2021},
  organization={IEEE}
}

@article{chen2024m,
  title={M $\^{} 3$ AV: A Multimodal, Multigenre, and Multipurpose Audio-Visual Academic Lecture Dataset},
  author={Chen, Zhe and Liu, Heyang and Yu, Wenyi and Sun, Guangzhi and Liu, Hongcheng and Wu, Ji and Zhang, Chao and Wang, Yu and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2403.14168},
  year={2024}
}
```