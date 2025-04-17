import torch
from model import SharedModel

if torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Mac M1/M2
else:
    device = torch.device("cpu")  # CPU

tf_param = {
    'model_name': 'data/ckpt/roberta-base'
    # 'model_name': 'roberta-base'
}

mf_param = {
    'checkpoint_file': 'data/ckpt/checkpoint_last_musicbert_base_w_genre_head.pt',
    'data_path': 'data/midi_oct',
    'user_dir': 'model/musicbert',
}

text_label_dict = {rel: i for i, rel in enumerate(['BEFORE', 'AFTER', 'IS_INCLUDED', 'SIMULTANEOUS'])}
# text_label_dict.update({i:rel for i, rel in enumerate(['BEFORE', 'AFTER', 'IS_INCLUDED', 'SIMULTANEOUS'])})

music_label_dict = {rel: i for i, rel in enumerate(['before', 'after', 'is_included', 'simultaneous'])}
# music_label_dict.update({i:rel for i, rel in enumerate(['before', 'after', 'is_included', 'simultaneous'])})


# run data/text/helper.py to get the following dict
sample_dict = {
    'train': 78018,
    'valid': 9753,
    'test': 9753
}

# music_label_dict = {rel:i for i, rel in enumerate(["before", "meets", "overlaps", "starts", "during", "finishes", "equals"])}
# music_label_dict.update({i:rel for i, rel in enumerate(["before", "meets", "overlaps", "starts", "during", "finishes", "equals"])})
        
global_model = SharedModel(
        hidden_dim      = 768,
        num_heads       = 8,
        num_layers      = 4,
        text_num_classes= 4,
        music_num_classes=4,
        dropout_rate    = 0.2
    )