import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', type=str, required=True)

args = parser.parse_args()

with open(args.json_dir, 'r') as file:
    data = json.load(file)

wav_id = args.json_dir.split('/')[-3]
index = 0

for entry in data:
    entry.pop('words_spoken', None)
    entry.pop('words_written', None)

output_file = args.json_dir.replace('final+timestamps.json','metadata.json')

with open(output_file, 'w') as file:
    json.dump(data, file, indent=2)
