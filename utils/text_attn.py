import matplotlib.pyplot as plt
import seaborn as sns
import torch
import matplotlib.colors as mcolors
from matplotlib import cm
from config import global_model, device 
import re
from torch.utils.data import DataLoader, default_collate
from dataset import TextDataset
import os
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import to_rgba, to_rgb
import json


def attn_pair_token(attns, token_pos, token_a, token_b):


    results = []
    try:
        token_indices_a = token_pos[token_a]
        token_indices_b = token_pos[token_b]
    except KeyError as e:
        print(f"Token '{e.args[0]}' not found in token_pos.")
        return pd.DataFrame()
    
    new_token_ids = []
    for token_span in token_indices_a:
        new_token_ids += token_span
    token_indices_a = new_token_ids
    new_token_ids = []
    for token_span in token_indices_b:
        new_token_ids += token_span
    token_indices_b = new_token_ids

    # print(token_indices_a)
    # print(token_indices_b)
    for layer_idx, layer_attn in enumerate(attns):  # [1, num_heads, seq_len, seq_len]
        attn = layer_attn[0]  # [num_heads, seq_len, seq_len]
        num_heads = attn.shape[0]

        for head_idx in range(num_heads):
            for qa in token_indices_a:
                for kb in token_indices_b:
                    if qa == kb:
                        continue
                    score_ab = attn[head_idx, qa, kb].item()
                    results.append({
                        "layer": layer_idx,
                        "head": head_idx,
                        "query": token_a,
                        "query_pos": qa,
                        "key": token_b,
                        "key_pos": kb,
                        "direction": f"{token_a}@{qa} → {token_b}@{kb}",
                        "attn_score": score_ab
                    })
                    score_ba = attn[head_idx, kb, qa].item()
                    results.append({
                        "layer": layer_idx,
                        "head": head_idx,
                        "query": token_b,
                        "query_pos": kb,
                        "key": token_a,
                        "key_pos": qa,
                        "direction": f"{token_b}@{kb} → {token_a}@{qa}",
                        "attn_score": score_ba
                    })

    df = pd.DataFrame(results)
    return df

def get_colormap_palette(name='tab10', n_colors=8, min_alpha=0.5):
    try:
        cmap = cm.get_cmap(name, n_colors)
        base_colors = [cmap(i) for i in range(n_colors)]
    except ValueError:
        base_colors = sns.color_palette(name, n_colors)
    
    alphas = [min_alpha + (1.0 - min_alpha) * (i / (n_colors - 1)) for i in range(n_colors)]
    rgba_palette = [to_rgba(to_rgb(color), alpha=a) for color, a in zip(base_colors, alphas)]
    return rgba_palette

def visualize_attention_trends(df: pd.DataFrame, save_dir="./", selected_directions=None):
    

    os.makedirs(save_dir, exist_ok=True)

    df['layer'] = df['layer'].astype(int)
    df['head'] = df['head'].astype(int)
    
    
    
    avg_df = df.groupby(['epoch', 'layer', 'head']).agg({
        'attn_score': 'mean'
    }).reset_index()
    # print(avg_df)
    
    
    topk_df = df.groupby(['layer', 'head']).agg({
        'attn_score': 'mean'
    }).reset_index()
    topk_df = topk_df.sort_values(by='attn_score', ascending=False)
    topk_df = topk_df.head(5)
    topk_df = topk_df[['layer', 'head']].values.tolist()
    # topk_df = [[f"L{l}", f"H{h}"] for l, h in topk_df]
    topk_df = [f"{l}-{h}" for l, h in topk_df]
    topk_head = ', '.join(topk_df)
    
    
    head_palette = get_colormap_palette(n_colors=8, min_alpha=1)
    layer_styles = [':', '-.', '--', '-']
    plt.figure(figsize=(14, 6))

    for (layer, head), group in avg_df.groupby(['layer', 'head']):
        color = head_palette[head]
        linestyle = layer_styles[layer]
        sns.lineplot(
            data=group,
            x='epoch',
            y='attn_score',
            color=color,
            linestyle=linestyle,
            linewidth=2,
            alpha=color[-1],
            markers=True
        )
        sns.scatterplot(
            data=group,
            x='epoch',
            y='attn_score',
            color=color,
            s=40,
            edgecolor='black',
            alpha=0.8
        )

    from matplotlib.lines import Line2D

    layer_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle=style, label=f'L{l}')
        for l, style in enumerate(layer_styles)
    ]

    head_handles = [
        Line2D([0], [0], color=head_palette[h], lw=3, label=f'H{h}')
        for h in range(len(head_palette))
    ]

    title_layer = Line2D([], [], linestyle='none', label='Layer', color='black')
    title_head = Line2D([], [], linestyle='none', label='Head', color='black')

    all_handles = [title_layer] + layer_handles + [title_head] + head_handles

    plt.legend(
        handles=all_handles,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        title=None,
    )
    plt.title(f'Avg Attention Score Between Event/Time over Epochs (Text, Label:{save_dir.split("/")[-1]})')
    plt.ylabel("Avg Attn Score (A<>B)")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.grid('both')
    plt.savefig(f"{save_dir}/avg_direction_epoch_text.png")
    plt.close()
    print(f"Avg attention plot saved to: {save_dir}/avg_direction_epoch_text.png")



    print(f"Attention plots saved to: {save_dir}")
    return topk_head

def find_token_pos(matches, tokens):

    def find_token_span_by_sequence(tokens, phrase_tokens):
      
        spans = []
        L = len(phrase_tokens)
        for i in range(len(tokens) - L + 1):
            if tokens[i:i+L] == phrase_tokens:
                spans.append(list(range(i, i+L)))
        return spans
    event_spans = {}
    for ev in matches:
        ev_tokens = global_model.text_encoder.tokenizer.tokenize(ev)
        spans = find_token_span_by_sequence(tokens, ev_tokens)
        event_spans[ev] = spans

    # print("Matched token spans:")
    # for ev, spans in event_spans.items():
    #     print(f"Event '{ev}' → Token spans: {spans}")
    return event_spans

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)



if __name__ == "__main__":
    labels = ['BEFORE', 'AFTER', 'IS_INCLUDED', 'SIMULTANEOUS']
    ckpt_dir = 'train_logs/ckpt/best'
    batch_size = 10
    for label in labels:
        result_dir = os.path.join('text_attn', ckpt_dir.split('/')[-1], label)
        os.makedirs(result_dir, exist_ok=True)
        print(f"Processing label: {label}")
        text_dataset = TextDataset(data_dir='data/text', set_name='test', sample_size=250, max_length=512, specified_label=label)
        text_dataset.save_data_list(os.path.join(result_dir, 'sample_data.csv'))
        text_loader = DataLoader(
            text_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: safe_collate(batch)
        )
        print('Loaded dataset')
        tokenizer = global_model.text_encoder.tokenizer
        model = global_model
        model.to(device)
        model.eval()
        result_df = pd.DataFrame()
        for epoch in tqdm(range(1, 52)):
            ckpt_path = os.path.join(ckpt_dir, f'checkpoints_epoch{epoch}.pth')
            print(f"Loading checkpoint from {ckpt_path}")
            model.load_weights(ckpt_path)
            for i, batch in enumerate(text_loader):
                if batch is None:
                    continue
                input_ids, attention_mask, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                input_data = {'input_ids': input_ids, 'attention_mask': attention_mask}

                with torch.no_grad():
                    emb, attns = model(input_data, type='text', return_trsfm_embedding=True, return_attns=True)

                for si in range(len(label)):
                    cur_attns = [attn_layer[si, :, :, :].unsqueeze(0) for attn_layer in attns]
                    input_ids = input_data['input_ids'][si]  # shape: [seq_len]
                    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
                    for sentence_len, token in enumerate(tokens):
                        if token == '<pad>':
                            break
                    tokens = tokens[:sentence_len]
                    # print(f"Input tokens: {tokens}")
                    
                    
                    text = tokenizer.decode(input_ids, skip_special_tokens=True)
                    # print(text)
                    question = text.split('What')[-1]
                    # print(question)
                    matches = re.findall(r"'(.*?)'", question)
                    # print(matches)
                    
                    token_pos = find_token_pos(matches, tokens)
                    df = attn_pair_token(cur_attns, token_pos, matches[0], matches[1])
                    df['epoch'] = epoch
                    df['sample_index'] = batch_size * i + si
                    # print(df)
                    result_df = pd.concat([result_df, df], ignore_index=True)

        result_df.to_csv(os.path.join(result_dir, 'attention_statistics.csv'), index=False)
        
    all_df = pd.DataFrame()
    topk_heads = {}
    for label in tqdm(labels):
        result_dir = os.path.join('text_attn', ckpt_dir.split('/')[-1], label)
        if not os.path.isdir(result_dir):
            continue
        print(f"Processing label: {label}")
        if not os.path.exists(os.path.join(result_dir, 'attention_statistics.csv')):
            print(f"File not found: {os.path.join(result_dir, 'attention_statistics.csv')}")
            continue
        result_df = pd.read_csv(os.path.join(result_dir, 'attention_statistics.csv'))
        topk_head = visualize_attention_trends(result_df, save_dir=result_dir, selected_directions=None)
        topk_heads[label] = topk_head
        all_df = pd.concat([all_df, result_df], ignore_index=True)
    result_dir = os.path.join('text_attn', ckpt_dir.split('/')[-1], 'ALL')
    os.makedirs(result_dir, exist_ok=True)
    all_df.to_csv(os.path.join(result_dir, 'attention_statistics.csv'), index=False)
    topk_head = visualize_attention_trends(all_df, save_dir=result_dir, selected_directions=None)
    topk_heads['ALL'] = topk_head
    with open(os.path.join(result_dir, 'topk_heads.json'), 'w') as f:
        json.dump(topk_heads, f, indent=4)
    print(f"Attention plots saved to: {result_dir}")