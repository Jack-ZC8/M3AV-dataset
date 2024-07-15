#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=5000
data_dir="data"

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -z "${M3AV}" ]; then
    echo "M3AV dataset was or will be saved to ${M3AV}"
else
    log "Fill the value of 'M3AV' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
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
         exit 1
      fi
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  if [ ! -d ${data_dir} ]; then
    mkdir ${data_dir}
  fi
  json_file="local/set_split.json"
  for dset in train dev test; do
      if [ ! -d ${data_dir}/${dset} ]; then
          mkdir ${data_dir}/${dset}
      fi
      ids=$(grep -E "\"${dset}\"" "$json_file" | awk -F'"' '{print $2}')
      for id in $ids; do
          echo "${id} ${M3AV}/dataset_v1.0_onlyaudio/${id}.flac" >> ${data_dir}/${dset}/wav.scp
          python3 local/data_prep.py --json_file ${M3AV}/dataset_v1.0_noaudio/${id}/speech --wav_file ${M3AV}/dataset_v1.0_onlyaudio/${id}.flac --output_path ${data_dir}/${dset}
      done
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  for dset in train dev test; do
    utils/utt2spk_to_spk2utt.pl data/${dset}/utt2spk >> data/${dset}/spk2utt
  done
fi