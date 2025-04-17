import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import random
import json
import torch
from torch.utils.data.dataloader import default_collate
import pandas as pd
from torch.utils.data import Dataset
from config import global_model, text_label_dict, music_label_dict, device, sample_dict

class TextDataset(Dataset):
    def __init__(self, data_dir='data/text', set_name='train', 
                sample_size=None, max_length=512, relations=text_label_dict,
                max_sample=sample_dict, text_csv=None, specified_label=None):
        if text_csv:
            self.data_list = pd.read_csv(text_csv)
        else:
            data_path = os.path.join(data_dir, f"{set_name}_data.csv")
            self.data_list = pd.read_csv(data_path)

        self.tokenizer = global_model.text_encoder.tokenizer
        self.max_length = max_length
        self.relations = relations
        num_labels = len(relations.keys())
        
        if specified_label:
            self.data_list = self.data_list[self.data_list['Answer'] == specified_label]
            
        if sample_size:
            if max_sample and sample_size > max_sample[set_name]:
                sample_size = max_sample[set_name] 
                
            groups = self.data_list.groupby('Answer')
            num_labels = len(groups)
            per_class = sample_size // num_labels

            sampled = [group.sample(n=min(per_class, len(group)), random_state=42)
                    for _, group in groups]
            sampled_df = pd.concat(sampled)

            current_size = len(sampled_df)
            shortfall = sample_size - current_size

            if shortfall > 0:
                remainder = self.data_list.drop(sampled_df.index)
                extra = remainder.sample(n=min(shortfall, len(remainder)), random_state=42)
                sampled_df = pd.concat([sampled_df, extra])

            self.data_list = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
                

    def save_data_list(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data_list.to_csv(path, index=False)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list.iloc[idx]
        text, label = data['Question'], data['Answer']

        # Tokenization + Padding + Truncation
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        if not encoded_text["input_ids"].shape[1] == self.max_length:
            # print(f"Text length unmatched: {encoded_text['input_ids'].shape[1]} - {self.max_length}")
            return None
        
        label = torch.tensor(self.relations[label], dtype=torch.long)

        return encoded_text["input_ids"].squeeze(0), encoded_text["attention_mask"].squeeze(0), label



class MusicDataset(Dataset):
    def __init__(self, data_dir='data/midi_oct', set_name='train', 
                random_pair=False, sample_size=None,
                max_length = 1024,
                relations=music_label_dict,
                max_sample = sample_dict,
                txt_list_json = None,
                specified_label=None):
        
        self.data_dir = os.path.join(data_dir, set_name)
        if txt_list_json:
            self.txt_list = json.load(open(txt_list_json, 'r'))
        else:
            self.txt_list = [
                os.path.join(root, f)
                for root, _, files in os.walk(self.data_dir) for f in files
                if os.path.isfile(os.path.join(root, f))
            ]
        random.shuffle(self.txt_list)
        self.random_pair = random_pair
        self.relations = relations
        self.max_length = max_length
        num_labels = len(relations.keys())
        if specified_label:
            self.txt_list = [txt_file for txt_file in self.txt_list if specified_label in txt_file]
            num_labels = 1
        if sample_size:
            if max_sample and sample_size > max_sample[set_name]:
                sample_size = max_sample[set_name] 
            per_class_sample = sample_size // num_labels
            
            label_to_files = {label: [] for label in relations.keys()}
            for txt_file in self.txt_list:
                label_name = os.path.basename(os.path.dirname(txt_file))  # same as txt_file.split('/')[-2]
                if label_name in label_to_files:
                    label_to_files[label_name].append(txt_file)

            sampled_txt_list = []
            for label_name, files in label_to_files.items():
                n = min(len(files), per_class_sample)
                sampled = random.sample(files, n)
                sampled_txt_list.extend(sampled)

            self.txt_list = sampled_txt_list

    def save_data_list(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.txt_list, f, indent=4)
    
    def __len__(self):
        return len(self.txt_list)


    def truncate_center(self, tokenized_text, keep_bound_length=None):

        seq_length = len(tokenized_text)
        if seq_length <= self.max_length:
            return tokenized_text
        
        if keep_bound_length is None: keep_bound_length = min(seq_length // 2, self.max_length // 3)

        num_tokens_to_remove = seq_length - self.max_length

        prefix = tokenized_text[:keep_bound_length]
        suffix = tokenized_text[-keep_bound_length:]
        center = tokenized_text[keep_bound_length:-keep_bound_length]
        keep_indices = random.choices(range(len(center)), k=len(center) - num_tokens_to_remove)
        center = center[keep_indices]
        truncated_text = torch.cat([prefix, center, suffix], dim=0)
        return truncated_text

    def padded(self, tensor, musicbert):
        if tensor.size(0) > self.max_length:
            return self.truncate_center(tensor)
        pad_length = self.max_length - tensor.size(0)
        return torch.cat([tensor, torch.full((pad_length,), musicbert.task.source_dictionary.pad(), dtype=torch.long)])


    def __getitem__(self, idx):
        txt_file = self.txt_list[idx]
        with open(txt_file, 'r', encoding='utf-8') as f:
            oct_pair = [line.strip() for line in f.readlines() if line.strip()]
        label = torch.tensor(self.relations[txt_file.split('/')[-2]])
        musicbert = global_model.music_encoder.musicbert
        musicbert.eval()
        
        tokenized_texts = [musicbert.task.source_dictionary.encode_line(line) for line in oct_pair]
        if self.random_pair:
            random.shuffle(tokenized_texts)
        try:
            oct_pair = torch.stack([self.padded(t, musicbert) for t in tokenized_texts])
        except Exception as e:
            print(f"truncate/pad error: {e}")
            return None
        if any(len(t) > self.max_length for t in oct_pair):
                return None 
        return oct_pair, label






if __name__ == '__main__':
    train_dataset_music = MusicDataset(data_dir='data/midi_oct', set_name='test')
    train_loader_music = DataLoader(train_dataset_music, batch_size=32, shuffle=True, collate_fn=lambda batch: default_collate([b for b in batch if b is not None]))
    train_dataset_text = TextDataset(data_dir='data/text', set_name='test')
    train_loader_text = DataLoader(train_dataset_text, batch_size=32, shuffle=True, collate_fn=lambda batch: default_collate([b for b in batch if b is not None]))

    model = global_model
    
    for batch_idx, (xt_yt, xm_ym) in enumerate(zip(train_loader_text, train_loader_music)):
        print(f"Batch {batch_idx + 1}")
        xti, xta, yt = xt_yt[0].to(device), xt_yt[1].to(device), xt_yt[2].to(device)
        xt = {
            'input_ids': xti,
            'attention_mask': xta
        }
        xm, ym = xm_ym[0].to(device), xm_ym[1].to(device)
        print("X: ", xti.shape, xta.shape, xm.shape)
        print("Labels:", yt, ym)
        model = model.to(device)
        model.train()
        
        logits_text = model(xt, type='text')
        logits_music = model(xm, type='music')
        print(logits_text, logits_music)