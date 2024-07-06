#!/usr/bin/env bash

set -e
set -u
set -o pipefail

#M3AV=/PATH/TO/M3AV

M3AV=/GPFS/data/heyangliu-1/dataset/M3AV-dataset-test

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ! -z "${M3AV}" ]; then
    echo "M3AV dataset path: ${M3AV}"
fi

if [ -d "${M3AV}/dataset_v1.0_noaudio" ]; then
  echo "M3AV transcriptions found."
else
  echo "M3AV transcriptions not found in ${M3AV}."
  echo "Please download dataset_v1.0_noaudio first."
fi

if [ -d "${M3AV}/dataset_v1.0_onlyaudio" ]; then
  echo "M3AV audio found."
else
  echo "M3AV audio not found in ${M3AV}."
  if [ -d "${M3AV}/dataset_v1.0_noaudio" ]; then
     echo "Download M3AV audio begin."
     mkdir ${M3AV}/dataset_v1.0_onlyaudio
     for dir in ${M3AV}/dataset_v1.0_noaudio/*; do
       yt-dlp -x --audio-format flac -o ${M3AV}/dataset_v1.0_onlyaudio/`basename $dir .flac` `cat $dir/raw/*.ytbUrl`
     done
     echo "Download M3AV audio finish."
  else
     echo "Please download dataset_v1.0_noaudio first."
  fi
  echo "Please download dataset_v1.0_noaudio first."
fi

if [ -e "${M3AV}/dataset_v1.0_noaudio/CHI-003EC/speech/metadata.json" ]; then
  echo "M3AV metadata preparation already done."
else
  echo "Prepare M3AV metadata"
  shopt -s globstar
  for file in ${M3AV}/dataset_v1.0_noaudio/**/speech/final+timestamps.json; do
    python3 utils/form_metadata.py --json_dir $file
  done
fi

if [ -e "${M3AV}/merged_metadata.json" ]; then
  echo "M3AV metadata already merged"
else
  echo "Merge M3AV metadata"
  python3 utils/merge_json.py --json_dir ${M3AV}/dataset_v1.0_noaudio
fi