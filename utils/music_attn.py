import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import to_rgba, to_rgb
from matplotlib import cm
from config import global_model, device 
from torch.utils.data import DataLoader, default_collate
from dataset import MusicDataset
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

oct_range = {0: 3, 1: 259, 2: 387, 3: 516, 4: 772, 5: 900, 6: 932, 7: 1186, 8: 1235, 9: 1236}
attr_list = ['Special', 'TS', 'BPM', 'Bar', 'Pos', 'Inst', 'Pitch', 'Dur', 'Vel', 'Mask']
oct_dict = {i: attr for i, attr in enumerate(attr_list)}

def get_attn_cross_segment(attn, split_index=None):

    if not split_index: split_index = attn.shape[3] // 2
    A2B = attn[:, :, :split_index, split_index:]
    B2A = attn[:, :, split_index:, :split_index]

    A2B = torch.mean(A2B, dim=(-1, -2))
    B2A = torch.mean(B2A, dim=(-1, -2))
    records = []
    for layer in range(attn.shape[0]):
        for head in range(attn.shape[1]):
            records.append({

                "layer": layer,
                "head": head,
                "A2B": float(A2B[layer][head]),
                "B2A": float(B2A[layer][head]),
                "AB" : float(A2B[layer][head] + B2A[layer][head]),
            })
    df = pd.DataFrame(records)
    return df
    

def get_attn_cross_token(attn: torch.Tensor, tokens) -> pd.DataFrame:

    tokens_np = np.array(tokens.cpu()).flatten()  # shape: [seq_len]
    seq_len = tokens_np.shape[0]

    sorted_items = sorted(oct_range.items(), key=lambda kv: kv[1])
    thresholds = np.array([item[1] for item in sorted_items])  # shape: [n_attr]
    attr_labels = np.array([item[0] for item in sorted_items])   # shape: [n_attr]

    mask = tokens_np[:, None] <= thresholds[None, :]  # shape: [seq_len, n_attr]
    attr_candidates = np.where(mask, attr_labels[None, :], np.inf)
    attributes = np.min(attr_candidates, axis=1)  # shape: [seq_len]

    attn = torch.mean(attn, dim=-2)
    L, H, _ = attn.shape
    attn_np = attn.cpu().numpy()  # shape: [L, H, seq_len]

    layer_arr = np.repeat(np.arange(L)[:, None, None], H, axis=1)  # shape: [L, H, 1] -> repeat along seq_len
    layer_arr = np.repeat(layer_arr, seq_len, axis=2)                # shape: [L, H, seq_len]

    head_arr = np.tile(np.arange(H)[None, :, None], (L, 1, seq_len))   # shape: [L, H, seq_len]
    pos_arr = np.tile(np.arange(seq_len)[None, None, :], (L, H, 1))      # shape: [L, H, seq_len]

    layer_flat = layer_arr.flatten()
    head_flat = head_arr.flatten()
    pos_flat = pos_arr.flatten()
    attn_flat = attn_np.reshape(-1)  # shape: [L*H*seq_len]

    tokens_flat = np.tile(tokens_np, L * H)
    attributes_flat = np.tile(attributes, L * H)

    df = pd.DataFrame({
        "layer": layer_flat,
        "head": head_flat,
        "position": pos_flat,
        "token": tokens_flat,
        "attribute": attributes_flat,
        "avg_attention": attn_flat
    })

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

def plot_attn_cross_segment(df, save_dir="./"):
    """Plot attention cross segment.
    df columns: layer,head,A2B,B2A,AB,epoch,sample_index
    the trends of attn score of A2B, B2A, AB in each layer-head following the epoch
    """
    
    
    os.makedirs(save_dir, exist_ok=True)

    df['layer'] = df['layer'].astype(int)
    df['head'] = df['head'].astype(int)
    topk_head = {col: [] for col in ['A2B', 'B2A', 'AB']}
    
    for col in topk_head.keys():
        topk_df = df.groupby(['layer', 'head']).agg({
            col: 'mean'
        }).reset_index()
        topk_df = topk_df.sort_values(by=col, ascending=False)
        topk_df = topk_df.head(5)
        topk_df = topk_df[['layer', 'head']].values.tolist()
        # topk_df = [[f"L{l}", f"H{h}"] for l, h in topk_df]
        topk_df = [f"{l}-{h}" for l, h in topk_df]
        topk_df = ', '.join(topk_df)
        topk_head[col] = topk_df

            
    # Figure 1: Top 5 heads for A2B, B2A, AB
    for col in ['A2B', 'B2A', 'AB']:
        avg_df = df.groupby(['epoch', 'layer', 'head']).agg({
            col: 'mean'
        }).reset_index()
        head_palette = get_colormap_palette(n_colors=8, min_alpha=1)
        layer_styles = [':', '-.', '--', '-']
        plt.figure(figsize=(14, 6))

        for (layer, head), group in avg_df.groupby(['layer', 'head']):
            color = head_palette[head]
            linestyle = layer_styles[layer]
            
            sns.lineplot(
                data=group,
                x='epoch',
                y=col,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                alpha=color[-1],
                markers=True
            )
            sns.scatterplot(
                data=group,
                x='epoch',
                y=col,
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
        plt.title('Avg Attention Score Cross Segments over Epochs (Music)')
        plt.ylabel(f"Avg Attn Score ({col})")
        plt.xlabel("Epoch")
        plt.grid('both')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/avg_cross_seg_epoch_music_{col}.png")
        plt.close()
        print(f"Avg attention plot saved to: {save_dir}/avg_cross_seg_epoch_music_{col}.png")


    print(f"Attention plots saved to: {save_dir}")
    return topk_head




def plot_attn_cross_token(df, save_dir="./"):
    """Plot attention cross tokens.
    df columns: layer,head,position,token,attribute,avg_attention,epoch,sample_index
    the trends of attn score of attribute, B2A, AB in each layer-head following the epoch
    """
    
    
    os.makedirs(save_dir, exist_ok=True)

    df['layer'] = df['layer'].astype(int)
    df['head'] = df['head'].astype(int)
    df = df.groupby(['epoch', 'layer', 'head', 'attribute']).agg({
        'avg_attention': 'mean'
    }).reset_index()
    
    
    topk_attr_heads = {}
    for attr, group in df.groupby('attribute'):
        topk_df = group.groupby(['layer', 'head']).agg({
            'avg_attention': 'mean'
        }).reset_index()
        topk_df = topk_df.sort_values(by='avg_attention', ascending=False).head(5)
        topk_df = topk_df[['layer', 'head']].values.tolist()
        # topk_df = [[f"L{l}", f"H{h}"] for l, h in topk_df]
        topk_df = [f"{l}-{h}" for l, h in topk_df]
        topk_df = ', '.join(topk_df)
        topk_attr_heads[attr] = topk_df
    
    
    # Figure 1: Which attribute is most attended by which layer/head over each epoch
    df['rank'] = df.groupby(['epoch', 'layer', 'head'])['avg_attention'].rank("dense", ascending=False)
    df['rank'] = df['rank'].astype(int)
    df['attribute'] = df['attribute'].astype(int)
    top_attr_df = df[df['rank'] == 1].copy()

    top_attr_df = top_attr_df.sort_values(by=['epoch', 'layer', 'head', 'attribute'])
    top_attr_df = top_attr_df.drop_duplicates(subset=['epoch', 'layer', 'head'])
    head_palette = get_colormap_palette(n_colors=8, min_alpha=1)
    layer_styles = [':', '-.', '--', '-']
    plt.figure(figsize=(14, 6))

    for (layer, head), group in top_attr_df.groupby(['layer', 'head']):
        color = head_palette[head]
        linestyle = layer_styles[layer]
        sns.lineplot(
            data=group,
            x='epoch',
            y='attribute',
            color=color,
            linestyle=linestyle,
            linewidth=2,
            alpha=color[-1],
            markers=True
        )
        sns.scatterplot(
            data=group,
            x='epoch',
            y='attribute',
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
        bbox_to_anchor=(1.005, 1),
        loc='upper left',
        frameon=True,
        title=None,
    )
    plt.title('Attributes Attended Cross Tokens over Epochs (Music)')
    plt.ylabel(f"Attributes Most Attended")
    plt.xlabel("Epoch")
    plt.subplots_adjust(left=0.1)
    # plt.tight_layout(pad=5, h_pad=-2)
    plt.grid('both')
    plt.ylim((-1, 10))
    plt.yticks(range(10), attr_list)
    plt.savefig(f"{save_dir}/avg_cross_tokens_epoch_music_attr.png")
    plt.close()
    print(f"Avg attention plot saved to: {save_dir}/avg_cross_seg_epoch_music_attr.png")

    # Figure 2: Avg attn score of attributes by all layer and heads over epochs
    avg_df = df.groupby(['epoch', 'attribute']).agg({
        'avg_attention': 'mean'
    }).reset_index()
    avg_df = avg_df.sort_values(by=['epoch', 'attribute'])
    attr_palette = get_colormap_palette(n_colors=10, min_alpha=1)
    plt.figure(figsize=(14, 8))

    for attr, group in avg_df.groupby('attribute'):
        color = attr_palette[attr % len(attr_palette)]
        sns.lineplot(
            data=group,
            x='epoch',
            y='avg_attention',
            color=color,
            linewidth=2,
            alpha=color[-1],
            markers=True
        )
        sns.scatterplot(
            data=group,
            x='epoch',
            y='avg_attention',
            color=color,
            s=40,
            edgecolor='black',
            alpha=0.8 
        )
        
    plt.legend(
        handles=[Line2D([0], [0], color=attr_palette[i], lw=3, label=attr_list[i]) for i in range(10)],
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        title=None,
    )
    plt.title('Avg Attention Score of Attributes over Epochs (Music)')
    plt.ylabel(f"Avg Attn Score")
    plt.xlabel("Epoch")
    plt.grid('both')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_cross_tokens_epoch_music_attr_score.png")
    plt.close()
    print(f"Avg attention plot saved to: {save_dir}/avg_cross_tokens_epoch_music_attr_score.png")
    
    
    print(f"Attention plots saved to: {save_dir}")
    return topk_attr_heads, df



def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None 
    return default_collate(batch)

if __name__ == "__main__":
    labels = [i.lower() for i in ['BEFORE', 'AFTER', 'IS_INCLUDED', 'SIMULTANEOUS']]
    print(labels)

    ckpt_dir = 'train_logs/ckpt/best'
    batch_size = 16
    for label in labels:
        # break
        result_dir = os.path.join('music_attn', ckpt_dir.split('/')[-1], label)
        os.makedirs(result_dir, exist_ok=True)
        print(f"Processing label: {label}")
        music_dataset = MusicDataset(data_dir='data/midi_oct', set_name='test', sample_size=250, max_length=512, specified_label=label)
        music_dataset.save_data_list(os.path.join(result_dir, 'sample_data.csv'))
        music_loader = DataLoader(
            music_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: safe_collate(batch)
        )
        print('Loaded dataset')
        model = global_model
        model.to(device)
        model.eval()
        ab_df = pd.DataFrame()
        token_df = pd.DataFrame()
        for epoch in tqdm(range(1, 52)):
            ckpt_path = os.path.join(ckpt_dir, f'checkpoints_epoch{epoch}.pth')
            print(f"Loading checkpoint from {ckpt_path}")
            model.load_weights(ckpt_path)
            for i, batch in enumerate(music_loader):
                if batch is None:
                    continue
                xm, ym = batch[0].to(device), batch[1].to(device)

                with torch.no_grad():
                    emb, attns = model(xm, type='music', return_trsfm_embedding=True, return_attns=True)

                for si in range(len(ym)):
                    cur_attns = [attn_layer[si, :, :, :].unsqueeze(0) for attn_layer in attns]
                    cur_attns = torch.cat(cur_attns, dim=0)
                    ab_records = get_attn_cross_segment(cur_attns)
                    token_records = get_attn_cross_token(cur_attns, xm[si])
                    ab_records['epoch'] = epoch
                    token_records['epoch'] = epoch
                    ab_records['sample_index'] = batch_size * i + si
                    token_records['sample_index'] = batch_size * i + si
                    ab_df = pd.concat([ab_df, ab_records], ignore_index=True)
                    token_df = pd.concat([token_df, token_records], ignore_index=True)
        ab_df.to_csv(os.path.join(result_dir, 'attn_cross_segment.csv'), index=False)
        plot_attn_cross_segment(ab_df, result_dir)
        print(f"Attention cross segment data saved to {os.path.join(result_dir, 'attn_cross_segment.csv')}")
        token_df['attribute'] = token_df['attribute'].astype(int)
        token_df.to_csv(os.path.join(result_dir, 'attn_cross_token.csv'), index=False)
        plot_attn_cross_token(token_df, result_dir)
        print(f"Attention cross token data saved to {os.path.join(result_dir, 'attn_cross_token.csv')}")
    ab = False
    token = True
    if ab:
        all_ab_df = pd.DataFrame()
        topk_heads_ab = {}
    if token:
        all_token_df = pd.DataFrame()
        topk_heads_token = {}
    
    
    for label in tqdm(labels):
        result_dir = os.path.join('music_attn', ckpt_dir.split('/')[-1], label)
        if ab:
            ab_df = pd.read_csv(os.path.join(result_dir, 'attn_cross_segment.csv'))
            topk_head = plot_attn_cross_segment(ab_df, result_dir)
            topk_heads_ab[label] = topk_head
            ab_df['label'] = label
            all_ab_df = pd.concat([all_ab_df, ab_df], ignore_index=True)
        if token:
            print('reading token_df')
            token_df = pd.read_csv(os.path.join(result_dir, 'attn_cross_token.csv'))
            topk_attr_heads, token_df = plot_attn_cross_token(token_df, result_dir)
            print(topk_attr_heads)
            topk_heads_token[label] = topk_attr_heads
            token_df['label'] = label
            all_token_df = pd.concat([all_token_df, token_df], ignore_index=True)
    result_dir = os.path.join('music_attn', ckpt_dir.split('/')[-1], 'all')
    os.makedirs(result_dir, exist_ok=True)
    if ab:
        all_ab_df.to_csv(os.path.join(result_dir, 'attn_cross_segment.csv'), index=False)
        topk_head = plot_attn_cross_segment(all_ab_df, result_dir)
        topk_heads_ab['all'] = topk_head
        with open(os.path.join(result_dir, 'topk_head.json'), 'w') as f:
            json.dump(topk_heads_ab, f, indent=4)
        print(f"Top 5 heads for each label: {topk_heads_ab}")
        print(f"Attention cross segment data saved to {os.path.join(result_dir, 'attn_cross_segment.csv')}")
    if token:
        all_token_df.to_csv(os.path.join(result_dir, 'attn_cross_token.csv'), index=False)
        topk_attr_heads, _ = plot_attn_cross_token(all_token_df, result_dir)
        topk_heads_token['all'] = topk_attr_heads
        with open(os.path.join(result_dir, 'topk_attr_heads.json'), 'w') as f:
            json.dump(topk_heads_token, f, indent=4)
        print(f"Top 5 heads for each label: {topk_heads_token}")
        print(f"Attention cross token data saved to {os.path.join(result_dir, 'attn_cross_token.csv')}")
    print("All attention data saved.")
    