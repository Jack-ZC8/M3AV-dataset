import json
import os
import argparse

from pydub.utils import mediainfo


def read_json(json_file,wav_file,output_path):
    total_time = 0
    json_file = json_file+'/final+timestamps.json'
    with open(json_file, 'r') as file:
        data = json.load(file)
    wav_id = wav_file.split('/')[-1].replace('.flac','')
    index = 0
    reco2dur = open(output_path + '/reco2dur', 'a')
    segments = open(output_path+'/segments','a')
    text = open(output_path + '/text', 'a')
    utt2spk = open(output_path + '/utt2spk', 'a')
    real_time = open(output_path + '/realtime', 'a')

    audio_info = mediainfo(wav_file)
    duration_sec = float(audio_info.get('duration', 0))

    reco2dur.write(wav_id+' '+str(duration_sec)+'\n')


    for entry in data:
        start_time,end_time = entry["timestr"].split('_')
        start_time = float(int(start_time)/1000)
        end_time = float(int(end_time)/1000)
        utt_id = wav_id + '_'+str("%04d" % index)
        segments.write(utt_id+' '+wav_id + ' '+str('%.2f'% start_time) + ' ' + str('%.2f'% end_time)+'\n')
        total_time = total_time + end_time - start_time
        utt2spk.write(utt_id+' '+wav_id+'\n')
        entry_text = entry.get('final_spoken')
        text.write(utt_id+' '+entry_text+'\n')
        index = index + 1
    real_time.write(wav_id+' '+str('%.2f'% total_time)+'\n')


parser = argparse.ArgumentParser(description='读取json文件和对应的flac音频文件，生成语音识别训练所需要的数据文件')

# 添加参数
parser.add_argument('--json_file', type=str, help='输入原JSON文件的路径')
parser.add_argument('--wav_file', type=str, help='输入原wav文件的路径')
parser.add_argument('--output_path', type=str, help='输出新JSON文件的路径')

args = parser.parse_args()
read_json(args.json_file,args.wav_file,args.output_path)
