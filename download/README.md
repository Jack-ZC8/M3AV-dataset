# Download

## LICENSE

The M<sup>3</sup>AV dataset is available to download for non-commercial purposes under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). M<sup>3</sup>AV doesn't own the copyright of the videos, the copyright remains with the original owners of the videos.

## Steps

### 1. Download tar archive

- `set_split.json` denotes the division of the set according to the speech speakers which is detailed in the paper.

- `dataset_v1.0_noaudio.tar.gz` denotes the content of the M3AV_v1.0 dataset other than audio. **It is enough if you just want to view OCR images with annotations and speech transcription.**

- `dataset_v1.0_onlyaudio_tar_gz/dataset_v1.0_onlyaudio.tar.gz.*` denote the audio data of M3AV_v1.0 dataset.

We provide three optional download links, all of which yield the same results:

1. [HuggingFace Link](https://huggingface.co/datasets/CHHHH/M3AV_v1.0)
2. [BaiduNetdisk Link](https://pan.baidu.com/s/1TPU3o9aa5TSBJ_YZ2_91-Q?pwd=v5j6)
3. [ModelScope Link](https://www.modelscope.cn/datasets/cc2024A/M3AV_v1.0)

Please use sha256 to verify that the download is valid by comparing your sha256 values and [ours](./M3AV_v1.0_sha256.txt).

### 2. Extract the tar archive

#### For Linux

```bash
tar -xzvf dataset_v1.0_noaudio.tar.gz

cat dataset_v1.0_onlyaudio_tar_gz/* | tar -xzv
```

#### For Win

```bash
copy /b dataset_v1.0_onlyaudio_tar_gz/dataset_v1.0_onlyaudio.tar.gz.* dataset_v1.0_onlyaudio.tar.gz

# Then you can use commonly used unpacking software to extract `dataset_v1.0_noaudio.tar.gz` and `dataset_v1.0_onlyaudio.tar.gz`.
```
