# MQTTS
 - Implementation for the paper: A Vector Quantized Approach for Text to Speech Synthesis on Real-World Spontaneous Speech.
 - Training and evaluating using audio from M<sup>3</sup>AV.
 - Slices of synthesized speech are available [here](https://guttural-lunaria-5bf.notion.site/MQTTS-Result-d7546f1d073b4805822fe028a0466614?pvs=4)
## Setup the environment
1. Setup conda environment:
```
conda create --name mqtts python=3.9
conda activate mqtts
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
(Update) You may need to create an access token to use the speaker embedding of pyannote.
If that's the case follow the [pyannote repo](https://github.com/pyannote/pyannote-audio) and specify YOUR_OWN_TOKEN in data/QuantizeDataset.py. 

2. Download the pretrained phonemizer checkpoint
```
wget https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt
```

## Preprocess the dataset
1. For spontaneous TTS, both audio and transcription are needed. Download M<sup>3</sup>AV following [here](https://github.com/Jack-ZC8/M3AV-dataset/tree/main/download) and put "dataset_v1.0_noaudio" and "dataset_v1.0_onlyaudio" in the same directory.
```
# Specify the data path in data.sh and check.
./data.sh
```

2. Install [FFmpeg](https://ffmpeg.org), then
```
conda install ffmpeg=4.3=hf484d3e_0
conda update ffmpeg
```
3. Run python script for data preprocess. Replace ${M3AV} with your own data path.
```
python utils/preprocess.py --M3AV_speech_dir ${M3AV}/dataset_v1.0_onlyaudio --MAL_json_dir merged_metadata.json --outputdir datasets 
```

## Train the quantizer and inference
1. Train
```
cd quantizer/
python train.py --input_wavs_dir ../datasets/audios \
                --input_training_file ../datasets/training.txt \
                --input_validation_file ../datasets/validation.txt \
                --checkpoint_path ./checkpoints \
                --config config.json
```

2. Inference to get codes for training the second stage
```
python get_labels.py --input_json ../datasets/train.json \
                     --input_wav_dir ../datasets/audios \
                     --output_json ../datasets/train_q.json \
                     --checkpoint_file ./checkpoints/g_00600000
python get_labels.py --input_json ../datasets/dev.json \
                     --input_wav_dir ../datasets/audios \
                     --output_json ../datasets/dev_q.json \
                     --checkpoint_file ./checkpoints/g_00600000
```

## Train the transformer (below an example for the 100M version)
```
cd ..
mkdir ckpt
CUDA_VISIBLE_DEVICES=3 python train.py \
     --distributed \
     --saving_path ckpt/ \
     --sampledir logs/ \
     --vocoder_config_path quantizer/checkpoints/config.json \
     --vocoder_ckpt_path quantizer/checkpoints/g_00600000 \
     --datadir datasets/audios \
     --metapath datasets/train_q.json \
     --val_metapath datasets/dev_q.json \
     --use_repetition_token \
     --ar_layer 4 \
     --ar_ffd_size 1024 \
     --ar_hidden_size 256 \
     --ar_nheads 4 \
     --speaker_embed_dropout 0.05 \
     --enc_nlayers 6 \
     --dec_nlayers 6 \
     --ffd_size 3072 \
     --hidden_size 768 \
     --nheads 12 \
     --batch_size 200 \
     --precision bf16 \
     --training_step 800000 \
     --layer_norm_eps 1e-05
```
if the training progress was interrupted, continue from the newest checkpoint
```
CUDA_VISIBLE_DEVICES=3 python train.py \
     --distributed \
     --saving_path ckpt/ \
     --sampledir logs/ \
     --resume_checkpoint ckpt/epoch=75-step=459999.ckpt \
     --vocoder_config_path quantizer/checkpoints/config.json \
     --vocoder_ckpt_path quantizer/checkpoints/g_00600000 \
     --datadir datasets/audios \
     --metapath datasets/train_q.json \
     --val_metapath datasets/dev_q.json \
     --use_repetition_token \
     --ar_layer 4 \
     --ar_ffd_size 1024 \
     --ar_hidden_size 256 \
     --ar_nheads 4 \
     --speaker_embed_dropout 0.05 \
     --enc_nlayers 6 \
     --dec_nlayers 6 \
     --ffd_size 3072 \
     --hidden_size 768 \
     --nheads 12 \
     --batch_size 200 \
     --precision bf16 \
     --training_step 800000 \
     --layer_norm_eps 1e-05
```
You can view the progress using:
```
tensorboard --logdir logs/
```

## Run batched inference

You'll have to change `speaker_to_text.json`, it's just an example.

```
mkdir infer_samples
CUDA_VISIBLE_DEVICES=2 python infer.py \
    --phonemizer_dict_path en_us_cmudict_forward.pt \
    --model_path ckpt/last.ckpt \
    --config_path ckpt/config.json \
    --input_path speaker_to_text.json \
    --outputdir infer_samples \
    --batch_size 4 \
    --top_p 0.8 \
    --min_top_k 2 \
    --phone_context_window 3 \
    --clean_speech_prior
```

### Pretrained checkpoints

1. Quantizer (put it under `quantizer/checkpoints/`): [here](https://sjtueducn-my.sharepoint.com/:u:/g/personal/liuheyang_sjtu_edu_cn/EVyi4SEjkb5Es1irpP8cBHUBdY3cJGx6ncIhGaIJQ0aICQ?e=OGxB8f)

2. Transformer (100M version) (put it under `ckpt/`): [model](https://sjtueducn-my.sharepoint.com/:u:/g/personal/liuheyang_sjtu_edu_cn/EYjLEf16UkJLnS0sttqn1wsBcGEW3l4FJOMr_1-QGgS15w?e=a8VLmx), [config](https://sjtueducn-my.sharepoint.com/:u:/g/personal/liuheyang_sjtu_edu_cn/EecsKbUgONJDumBJh-qsxN4B9r6GHA89hX9OwBLTvNjt6g?e=JVHETp)

### Citation
If you find this code useful for your research, please cite the following papers
```
@inproceedings{chen2023vector,
  title={A vector quantized approach for text to speech synthesis on real-world spontaneous speech},
  author={Chen, Li-Wei and Watanabe, Shinji and Rudnicky, Alexander},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={11},
  pages={12644--12652},
  year={2023}
}

@article{chen2024m,
  title={M $\^{} 3$ AV: A Multimodal, Multigenre, and Multipurpose Audio-Visual Academic Lecture Dataset},
  author={Chen, Zhe and Liu, Heyang and Yu, Wenyi and Sun, Guangzhi and Liu, Hongcheng and Wu, Ji and Zhang, Chao and Wang, Yu and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2403.14168},
  year={2024}
}
```