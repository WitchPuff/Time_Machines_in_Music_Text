import json
import pandas as pd
import os


def music_dataset_overview(json_path='data/midi_oct/split_dict.json'):
    with open(json_path, 'r') as f:
        split_dict = json.load(f)
    for set, keys in split_dict.items():
        print(set, len(keys)*8)
    
def text_dataset_overview(data_dir = 'data/text'):
    file_name = ['test_data.csv', 'train_data.csv', 'valid_data.csv']
    
    for file in file_name:
        print(f'File: {file}')
        df = pd.read_csv(os.path.join(data_dir, file))
        count = df['Answer'].value_counts()
        print(count)
        print(sum(count.values))
        print()

def music_sample_overview(ckpt_dir):
    json_paths = ['test_data.json', 'train_data.json', 'valid_data.json']
    for json_path in json_paths:
        print(f'File: {json_path}')
        with open(os.path.join(ckpt_dir, 'music', json_path), 'r') as f:
            txt_list = json.load(f)
        txt_list = [os.path.basename(os.path.dirname(txt)) for txt in txt_list]
        txt_list = pd.Series(txt_list)
        print(txt_list.value_counts())
        print()

def text_sample_overview(ckpt_dir):
    file_name = ['test_data.csv', 'train_data.csv', 'valid_data.csv']
    
    for file in file_name:
        print(f'File: {file}')
        df = pd.read_csv(os.path.join(ckpt_dir, 'text', file))
        print(df['Answer'].value_counts())
        print()


print('===== Dataset overview =====', end='\n\n')
print('===== Music dataset overview =====', end='\n\n')
music_dataset_overview()
print('===== Text dataset overview =====', end='\n\n')
text_dataset_overview() 
# ckpt_dir = 'train_logs/ckpt/epochs-40_batch_size-32_text_max_length-512_music_max_length-1024_sample_size-8000_warmup_step-1000_decay_step-50000_lr-8e-05_weight_decay-1e-05_1742732214.0642514'
# print('===== Music dataset overview =====', end='\n\n')
# music_sample_overview(ckpt_dir)
# print('===== Text dataset overview =====', end='\n\n')
# text_sample_overview(ckpt_dir)
