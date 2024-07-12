import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', type=str, required=True)

args = parser.parse_args()


base_folder = args.json_dir 

output_file = "merged_metadata.json"

output_data = []
for root, dirs, files in os.walk(base_folder):
    for filename in files:
        if filename == "metadata.json":
            filepath = os.path.join(root, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                wav_id = root.split('/')[-2]  # 使用子文件夹名作为wav_id
                file_data = {"wav_id": wav_id, "segments": data}
                output_data.append(file_data)

with open(output_file, 'w') as file:
    json.dump(output_data, file, indent=2)

print("Merged metadata saved to", output_file)