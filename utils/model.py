import torch
import torch.nn as nn
from fairseq.models.roberta import RobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os


class TaskClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)



    
    
class SharedTransformerBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads, 
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, embeddings: torch.Tensor, return_attns=False):
        """
        Input: [batch_size, seq_len, hidden_dim]
        Output: [batch_size, hidden_dim] OR (pooled, attn_weights)
        """
        attn_weights_all = [] if return_attns else None

        for layer in self.layers:
            if return_attns:
                # run self_attn manually to get attn weights
                src2, attn_weights = layer.self_attn(
                    embeddings, embeddings, embeddings, 
                    need_weights=True,
                    average_attn_weights=False
                )
                embeddings = embeddings + layer.dropout1(src2)
                embeddings = layer.norm1(embeddings)
                src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(embeddings))))
                embeddings = embeddings + layer.dropout2(src2)
                embeddings = layer.norm2(embeddings)
                attn_weights_all.append(attn_weights)
            else:
                embeddings = layer(embeddings)

        pooled = self.pooling(embeddings.transpose(1, 2)).squeeze(-1)  # [batch_size, hidden_dim]

        if return_attns:
            return pooled, attn_weights_all
        return pooled

class MusicEncoder(nn.Module):

    def __init__(self,
                 checkpoint_file='data/ckpt/checkpoint_last_musicbert_base_w_genre_head.pt',
                 data_path='data/midi_oct',
                 user_dir='model/musicbert',
                 random_pair=False):
        super().__init__()
        self.musicbert = RobertaModel.from_pretrained(
            '.',
            checkpoint_file=checkpoint_file,
            data_name_or_path=data_path,
            user_dir=user_dir
        )
        # freeze the params of MusicBERT
        for param in self.musicbert.parameters():
            param.requires_grad = False
        self.random_pair = random_pair
        
    
        
    def forward(self, oct: torch.Tensor) -> torch.Tensor:

        features = self.musicbert.extract_features(oct)

        return features.detach()


class TextEncoder(nn.Module):
    def __init__(self, 
                 model_name="data/ckpt/roberta-base"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.model.eval()
    
        feature = self.model(**x, output_hidden_states=True)
        feature = feature.hidden_states[-1]

        # last_4_layers = torch.stack(hidden_states[-4:])  # (4, batch_size, seq_len, hidden_dim)
        # feature = torch.mean(last_4_layers, dim=0)
        
        return feature.detach()

class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_dim=768, dropout_rate=0.2):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class SharedModel(nn.Module):

    def __init__(self, mf_param: dict = None, tf_param: dict=None,
                    hidden_dim=768, num_heads=8, num_layers=4,
                    text_num_classes=4, music_num_classes=4, dropout_rate=0.2):
        super().__init__()
        self.music_encoder = MusicEncoder(**mf_param if mf_param else {})
        self.text_encoder = TextEncoder(**tf_param if tf_param else {})
        self.transformer_block = SharedTransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
        self.ffn = FeedForwardLayer()
        self.text_classifier = TaskClassifier(input_dim=hidden_dim, num_classes=text_num_classes)
        self.music_classifier = TaskClassifier(input_dim=hidden_dim, num_classes=music_num_classes)
        # self.dropout = nn.Dropout(p=dropout_rate)
        
    def save_weights(self, path):
        weights_to_save = {
            "transformer_block": self.transformer_block.state_dict(),
            "ffn": self.ffn.state_dict(),
            "text_classifier": self.text_classifier.state_dict(),
            "music_classifier": self.music_classifier.state_dict()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(weights_to_save, path)
        print(f"Weights saved to {path}.")

    def load_weights(self, path, strict=True):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))  # Ensure compatibility
        self.transformer_block.load_state_dict(checkpoint["transformer_block"], strict=strict)
        self.ffn.load_state_dict(checkpoint["ffn"], strict=strict)
        self.text_classifier.load_state_dict(checkpoint["text_classifier"], strict=strict)
        self.music_classifier.load_state_dict(checkpoint["music_classifier"], strict=strict)
        print(f"Weights loaded from {path}.")
    
    def forward(self, x, type='text', return_trsfm_embedding=False, return_ffn_embedding=False, return_attns=False):

        if type == 'music':
            x0 = self.music_encoder(x[:, 0, :])
            x1 = self.music_encoder(x[:, 1, :])
            x = torch.cat([x0, x1], dim=1)
        else:
            x = self.text_encoder(x)
            
        x = self.transformer_block(x, return_attns=return_attns)
        if return_trsfm_embedding: return x
        # x = self.dropout(x)
        x = self.ffn(x)
        if return_ffn_embedding: return x
        if type == 'music':
            y = self.music_classifier(x)
        else:
            y = self.text_classifier(x)

        return y
    

from collections import defaultdict

def count_parameters(model):
    print(f"{'Top-level Module':30} | {'#Params':>12} | {'Trainable':>9}")
    print("-" * 60)
    module_param_counts = defaultdict(lambda: [0, 0])  # [total, trainable]

    for name, param in model.named_parameters():
        top_level = name.split('.')[0]
        module_param_counts[top_level][0] += param.numel()
        if param.requires_grad:
            module_param_counts[top_level][1] += param.numel()

    total = 0
    total_trainable = 0
    for module, (count, trainable) in module_param_counts.items():
        total += count
        total_trainable += trainable
        print(f"{module:30} | {count:12,} | {str(trainable):>9}")
    print("-" * 60)
    print(f"{'Total':30} | {total:12,} |")
    print(f"{'Trainable Total':30} | {total_trainable:12,} |")


if __name__ == '__main__':
    model = SharedModel()
    print(model)
    count_parameters(model)
    # print(model.transformer_block.layers)
    # musicbert = model.music_encoder.musicbert
    # oct_dict = musicbert.task.target_dictionary.indices
    # oct_range = {0: 3}
    # for symbol, index in oct_dict.items():
    #     print(symbol, index)
    #     if symbol[1].isdigit():
    #         oct_range[int(symbol[1])] = index
    # print(oct_range)    
