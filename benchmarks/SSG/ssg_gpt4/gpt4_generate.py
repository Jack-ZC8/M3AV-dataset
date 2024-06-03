import copy
from glob import glob
import json
import os, random, time
import socket
import sys
import subprocess
from os.path import join as pjoin
from openai import OpenAI
import traceback
from tqdm import tqdm
from pprint import pprint
import base64
from mimetypes import guess_type

# Function to encode a local image into data URL
# https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest%2Csystem-assigned%2Cresource#call-the-chat-completion-apis
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


API_KEY = open('ssg_gpt4/openai_key.txt').read().strip()


def gpt4_generate(input_sentence, task, print_log=False):
    client = OpenAI(api_key=API_KEY)
    SYSTEM_PROMPT = {
        'speech2slide': "There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a slideshow creator and generate the text in the corresponding single slide based on the given speech text.",
        'slide2speech': "There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text in the given single slide."
    }
    completion = client.chat.completions.create(
        model='gpt-4',
        n=1,
        seed=0,
        top_p=0.9,
        temperature=1.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT[task]},
            {"role": "user", "content": input_sentence},
        ],
    )
    if print_log:
        print(completion)
        print('='*89)
    
    return completion.choices[0].message.content

def gpt4V_generate(image_url, print_log=False):
    client = OpenAI(api_key=API_KEY)
    try:
        completion = client.chat.completions.create(
            model='gpt-4-vision-preview',
            n=1,
            seed=0,
            top_p=0.9,
            temperature=1.0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the picture of the given single slide."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": image_url,},
                ],},
            ],
        )
    except:
        error_msg = traceback.format_exc()
        if 'content_policy_violation' in error_msg:
            return 'NULL'
        else:
            print(error_msg)
            exit(0)
    if print_log:
        print(completion)
        print('='*89)
    return completion.choices[0].message.content
    

if __name__ == "__main__":
    assert sys.argv[1] in ('gpt-4', 'gpt-4V')

    test_vid_list = [i.split('.')[0] for i in os.listdir('ssg_dataset/text_data/test/')]
        
    for task in [
        'speech2slide', 
        'slide2speech',
    ]:
        if sys.argv[1] == 'gpt-4V' and task == 'speech2slide':
            continue
        os.makedirs(pjoin('ssg_gpt4', sys.argv[1], task), exist_ok=True)
        for test_vid in tqdm(test_vid_list, desc=task):
            in_json_path = pjoin('ssg_dataset/text_data/test/', test_vid+'.json')
            output_path = pjoin('ssg_gpt4', sys.argv[1], task, os.path.basename(in_json_path))
            if os.path.exists(output_path):
                content = json.load(open(output_path, 'r', encoding='utf-8'))
            else:
                content = json.load(open(in_json_path, 'r', encoding='utf-8'))
            for unit in tqdm(content, desc=test_vid):
                output_key = 'generated_speech_text' if task == 'slide2speech' else 'generated_ocr_text'
                if output_key in unit:
                    continue
                if sys.argv[1] == 'gpt-4':
                    input_sentence = unit['speech_text'] if task == 'speech2slide' else unit['ocr_text']
                    output_sentence = gpt4_generate(
                        input_sentence=input_sentence,
                        task=task,
                        print_log=False
                    )
                else:
                    output_sentence = gpt4V_generate(
                        image_url=local_image_to_data_url(f"ssg_dataset/image_data/{unit['image_name']}"),
                        print_log=False
                    )
                unit[output_key] = output_sentence

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)

