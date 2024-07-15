#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/train_rnnt.yaml
inference_config=conf/decode_asr.yaml
asr_tag=train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 8 \
    --gpu_inference false \
    --inference_nj 16 \
    --nbpe 600 \
    --suffixbpe suffix \
    --max_wav_duration 30 \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --eval_valid_set true \
    --asr_tag ${asr_tag} \
    --inference_asr_model valid.loss.ave.pth \
    --biasing true \
    --bpe_train_text "data/${train_set}/text" "$@"
