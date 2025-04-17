import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba, to_rgb
from matplotlib import cm
from config import global_model, device 
from torch.utils.data import DataLoader, default_collate
from dataset import MusicDataset, TextDataset
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import json


def probing(music_train_data, music_train_label, music_test_data, music_test_label, 
            text_train_data, text_train_label, text_test_data, text_test_label,
            hidden_dim=768, method='zero'):
    """
        Train on both modalities, then test and patch data form both modalities
    """
    correct_music_patcheds, correct_text_patcheds = [], []
    music_train_data = music_train_data.cpu().numpy()
    music_train_label = music_train_label.cpu().numpy()
    music_test_data = music_test_data.cpu().numpy()
    music_test_label = music_test_label.cpu().numpy()
    text_train_data = text_train_data.cpu().numpy()
    text_train_label = text_train_label.cpu().numpy()
    text_test_data = text_test_data.cpu().numpy()
    text_test_label = text_test_label.cpu().numpy()
    
    train_data = np.concatenate((music_train_data, text_train_data), axis=0)
    train_label = np.concatenate((music_train_label, text_train_label), axis=0)

    indices = np.random.permutation(len(train_label))
    train_data = train_data[indices]
    train_label = train_label[indices]
    print("train_data.shape", train_data.shape)
    print("train_label.shape", train_label.shape)
    
    clf = LogisticRegression()
    clf.fit(train_data, train_label)

    preds_music_train_data = clf.predict(music_train_data)
    correct_music_train_data = np.sum(preds_music_train_data == music_train_label)


    preds_music_test_data = clf.predict(music_test_data)
    correct_music_test_data = np.sum(preds_music_test_data == music_test_label)
    
    preds_text_train_data = clf.predict(text_train_data)
    correct_text_train_data = np.sum(preds_text_train_data == text_train_label)
    
    preds_text_test_data = clf.predict(text_test_data)
    correct_text_test_data = np.sum(preds_text_test_data == text_test_label)
    

    for i in range(hidden_dim):
        # Patch music_test_data
        music_patched = patch_neuron(music_test_data, i, method=method)
        preds_music_patch = clf.predict(music_patched)
        correct_music_patch = np.sum(preds_music_patch == music_test_label)
        correct_music_patcheds.append(correct_music_patch)

        text_patched = patch_neuron(text_test_data, i, method=method)
        preds_text_patch = clf.predict(text_patched)
        correct_text_patch = np.sum(preds_text_patch == text_test_label)
        correct_text_patcheds.append(correct_text_patch)
        

    return (correct_music_train_data, correct_text_train_data), (correct_music_test_data, correct_music_patcheds), (correct_text_test_data, correct_text_patcheds)




def probing_gen(x_train, y_train, x_test, y_test, Z=None, Yz=None, hidden_dim=768, method_x='zero', method_z='src'):
    """
        Train on one modality, test on another modality to test generalization
    """
    correct_x_patcheds, correct_z_patcheds = [], []
    x_train = x_train.cpu().numpy()
    y_train = y_train.cpu().numpy()
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    preds_x_train = clf.predict(x_train)
    correct_x_train = np.sum(preds_x_train == y_train)


    preds_x_test = clf.predict(x_test)
    correct_x_test = np.sum(preds_x_test == y_test)
    
    

    correct_z = None
    if Z is not None and Yz is not None:
        Z = Z.cpu().numpy()
        Yz = Yz.cpu().numpy()
        preds_z = clf.predict(Z)
        correct_z = np.sum(preds_z == Yz)

    for i in range(hidden_dim):
        # Patch x_test
        X_patched = patch_neuron(x_test, i, method=method_x)
        preds_x_patch = clf.predict(X_patched)
        correct_x_patch = np.sum(preds_x_patch == y_test)
        correct_x_patcheds.append(correct_x_patch)

        if Z is not None and Yz is not None:
            # Patch Z using x_test as source
            Z_patched = patch_neuron(Z, i, method=method_z, src=x_test)
            preds_z_patch = clf.predict(Z_patched)
            correct_z_patch = np.sum(preds_z_patch == Yz)
            correct_z_patcheds.append(correct_z_patch)

    return correct_x_train, (correct_x_test, correct_x_patcheds), (correct_z, correct_z_patcheds)



def patch_neuron(x_train, neuron_idx, method='zero', src=None):
    x_patched = x_train.copy()
    if method == 'zero':
        x_patched[:, neuron_idx] = 0
    elif method == 'noise':
        x_patched[:, neuron_idx] = torch.randn_like(x_patched[:, neuron_idx])
    elif method == 'src':
        if src is None:
            raise ValueError("src must be provided when method is 'src'")
        x_patched[:, neuron_idx] = src[:, neuron_idx]
    return x_patched


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def extract_top_neurons(df: pd.DataFrame, save_dir="./", top_k=20):

    grouped = df.groupby("neuron_idx").mean(numeric_only=True)

    top_text = grouped["text_test"].nlargest(top_k).index.tolist()
    top_music = grouped["music_test"].nlargest(top_k).index.tolist()
    top_music_diff = (grouped["music_test"] - grouped["music_patched"]).nlargest(top_k).index.tolist()
    top_text_diff = (grouped["text_test"] - grouped["text_patched"]).nlargest(top_k).index.tolist()

    result = {
        "top_text": top_text,
        "top_music": top_music,
        "top_music_diff": top_music_diff,
        "top_text_diff": top_text_diff
    }


    
    with open(os.path.join(save_dir, 'top_neurons.json'), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Saved] Top neuron indices to {os.path.join(save_dir, 'top_neurons.json')}")
    print(result)    
    return result



def extract_top_neurons_gen(df_from_music: pd.DataFrame, df_from_text:pd.DataFrame, save_dir="./", top_k=10):
    result = {}
    grouped_from_music = df_from_music.groupby("neuron_idx").mean(numeric_only=True)
    grouped_from_text = df_from_text.groupby("neuron_idx").mean(numeric_only=True)

    top_text = grouped_from_music["text"].nlargest(top_k).index.tolist()
    top_music = grouped_from_music["music_test"].nlargest(top_k).index.tolist()
    top_music_diff = (grouped_from_music["music_test"] - grouped_from_music["music_patched"]).nlargest(top_k).index.tolist()
    top_text_diff = (grouped_from_music["text_patched"] - grouped_from_music["text"]).nlargest(top_k).index.tolist()

    ret = {
        "top_text": top_text,
        "top_music": top_music,
        "top_music_diff": top_music_diff,
        "top_text_diff": top_text_diff
    }

    result['from_music'] = ret
    
    top_text = grouped_from_text["text_test"].nlargest(top_k).index.tolist()
    top_music = grouped_from_text["music"].nlargest(top_k).index.tolist()
    top_music_diff = (grouped_from_text["music_patched"] - grouped_from_text["music"]).nlargest(top_k).index.tolist()
    top_text_diff = (grouped_from_text["text_test"] - grouped_from_text["text_patched"]).nlargest(top_k).index.tolist()

    ret = {
        "top_text": top_text,
        "top_music": top_music,
        "top_music_diff": top_music_diff,
        "top_text_diff": top_text_diff
    }

    result['from_text'] = ret

    with open(os.path.join(save_dir, 'top_neurons.json'), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Saved] Top neuron indices to {os.path.join(save_dir, 'top_neurons.json')}")
    print(result)    
    return result


    
def plot_accuracy_trends(df: pd.DataFrame, neuron_idx, save_dir="./"):

    if "music" in df.columns: 
        type = "from_text"
        cols = ["text_train", "text_test", "text_patched", "music", "music_patched"]
        text_keys = ["text_train", "text_test", "text", "text_patched"]
    elif 'text' in df.columns: 
        type = "from_music"
        cols = ["music_train", "music_test", "music_patched", "text", "text_patched"]
        music_keys = ["music_train", "music_test", "music", "music_patched"]
    else:
        cols = ['text_train', 'music_train', 'text_test', 'music_test', 'text_patched', 'music_patched']
        type = "both"
        text_keys = ["text_train", "text_test", "text_patched"]
        music_keys = ["music_train", "music_test", "music_patched"]
    
    df = df[df["neuron_idx"] == neuron_idx]
    df = df.drop(columns=["neuron_idx"])
    
    df_melted = df.melt(id_vars=["epoch"], 
                        value_vars=cols,
                        var_name="Type", 
                        value_name="Accuracy")

    palette = sns.color_palette("tab10", n_colors=len(cols))
    color_dict = {col: palette[i] for i, col in enumerate(cols)}
    
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 0.8)
    ax = sns.lineplot(data=df_melted, x="epoch", y="Accuracy", hue="Type", marker="o", palette=color_dict)
    plt.title(f"Accuracy Trends Over Epochs ({type}, Neuron: {neuron_idx})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    handles, labels = ax.get_legend_handles_labels()

    new_handles, new_labels = [], []

    def add_group(title, keys):
        new_handles.append(Line2D([], [], color='white', label=title))  # group title
        new_labels.append(title)
        for key in keys:
            if key in labels:
                idx = labels.index(key)
                new_handles.append(handles[idx])
                new_labels.append(labels[idx])

    add_group("- Text -", text_keys)

    add_group("- Music -", music_keys)

    ax.legend(new_handles, new_labels, title="Modality", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    
    plt.ylim(0, 0.8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"accuracy_trends_{type}_{neuron_idx}.png"))
    plt.close()
    print("[Saved] Accuracy trend plot:", os.path.join(save_dir, f"accuracy_trends_{type}_{neuron_idx}.png"))


def plot_patch_diff_trends(df: pd.DataFrame, neuron_idx, save_dir="./"):

    if "music" in df.columns:
        type = 'from_text'
        df["text_diff"] = df["text_patched"] - df["text_test"]
        df["music_diff"] = df["music_patched"] - df["music"]
    elif 'text' in df.columns:
        type = 'from_music'
        df["text_diff"] = df["text_patched"] - df["text"]
        df["music_diff"] = df["music_patched"] - df["music_test"]
    else:
        type = 'both'
        df["text_diff"] = df["text_patched"] - df["text_test"]
        df["music_diff"] = df["music_patched"] - df["music_test"]
        
    df = df[df["neuron_idx"] == neuron_idx]
    df = df.drop(columns=["neuron_idx"])
    df_melted = df.melt(id_vars=["epoch"], 
                        value_vars=["text_diff", "music_diff"],
                        var_name="Type", 
                        value_name="Delta Accuracy")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x="epoch", y="Delta Accuracy", hue="Type", marker="o")
    plt.title(f"Accuracy Change After Neuron Patching - {type} (Neuron: {neuron_idx})")
    plt.ylim(-0.01, 0.01)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Change (Patched - Original)")
    plt.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.legend(title="Modality")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"patch_diff_trends_{type}_{neuron_idx}.png"))
    plt.close()
    print("[Saved] Patch diff trend plot:", {os.path.join(save_dir, f"patch_diff_trends_{type}_{neuron_idx}.png")})

if __name__ == '__main__':
    text_labels = ['BEFORE', 'AFTER', 'IS_INCLUDED', 'SIMULTANEOUS']
    music_labels = [i.lower() for i in text_labels]
    ckpt_dir = 'train_logs/ckpt/best'
    batch_size = 16
    method_x = 'zero'
    mode = 'both'
    sample_size=1000
    epoch_max = 52
    result_dir = os.path.join('probing', ckpt_dir.split('/')[-1], 'all', method_x, mode)
    os.makedirs(result_dir, exist_ok=True)
    print('Loading dataset')
    train_music_dataset = MusicDataset(data_dir='data/midi_oct', set_name='valid', sample_size=sample_size, max_length=512)
    train_music_dataset.save_data_list(os.path.join(result_dir, 'sample_data_music_train.csv'))
    train_music_loader = DataLoader(
        train_music_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: safe_collate(batch)
    )
    train_text_dataset = TextDataset(data_dir='data/text', set_name='valid', sample_size=sample_size, max_length=512)
    train_text_dataset.save_data_list(os.path.join(result_dir, 'sample_data_text_train.csv'))
    train_text_loader = DataLoader(
        train_text_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: safe_collate(batch)
    )
    test_music_dataset = MusicDataset(data_dir='data/midi_oct', set_name='test', sample_size=sample_size, max_length=512)
    test_music_dataset.save_data_list(os.path.join(result_dir, 'sample_data_music_test.csv'))
    test_music_loader = DataLoader(
        test_music_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: safe_collate(batch)
    )
    test_text_dataset = TextDataset(data_dir='data/text', set_name='test', sample_size=sample_size, max_length=512)
    test_text_dataset.save_data_list(os.path.join(result_dir, 'sample_data_text_test.csv'))
    test_text_loader = DataLoader(
        test_text_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: safe_collate(batch)
    )
    
    print('Loaded dataset')
    model = global_model
    model.to(device)
    model.eval()
    all_from_text_df = pd.DataFrame()
    all_from_music_df = pd.DataFrame()
    all_df = pd.DataFrame()
    for epoch in tqdm(range(1, epoch_max)):
        ckpt_path = os.path.join(ckpt_dir, f'checkpoints_epoch{epoch}.pth')
        print(f"Loading checkpoint from {ckpt_path}")
        model.load_weights(ckpt_path)
        if mode == 'gen':
            total = {
                    'from_text': {
                        'text_train': 0,
                        'text_test': 0,
                        'music': 0,
                        'text_patched': [0] * 768,
                        'music_patched': [0] * 768
                    },
                    'from_music': {
                        'text': 0,
                        'music_train': 0,
                        'music_test': 0,
                        'text_patched': [0] * 768,
                        'music_patched': [0] * 768
                    }
            }
        elif mode == 'both':
            total = {
                    'text_train': 0,
                    'music_train': 0,
                    'text_test': 0,
                    'music_test': 0,
                    'text_patched': [0] * 768,
                    'music_patched': [0] * 768
                }

        text_sample = 0
        music_sample = 0
        for batch_idx, (xt_yt_train, xm_ym_train, xt_yt_test, xm_ym_test) in tqdm(
                enumerate(zip(train_text_loader, train_music_loader, test_text_loader, test_music_loader)),
                total=min(len(train_text_loader), len(train_music_loader), len(test_text_loader), len(test_music_loader)),
                desc="Batches"):
            # Unpack text data
            xti_train, xta_train, yt_train = xt_yt_train[0].to(device), xt_yt_train[1].to(device), xt_yt_train[2].to(device)
            xti_test, xta_test, yt_test = xt_yt_test[0].to(device), xt_yt_test[1].to(device), xt_yt_test[2].to(device)
            xm_train, ym_train = xm_ym_train[0].to(device), xm_ym_train[1].to(device)
            xm_test, ym_test = xm_ym_test[0].to(device), xm_ym_test[1].to(device)
            
            min_size = min(yt_train.size(0), ym_train.size(0), yt_test.size(0), ym_test.size(0))
            xti_train, xta_train, yt_train  = xti_train[:min_size], xta_train[:min_size], yt_train[:min_size]
            xti_test, xta_test, yt_test  = xti_test[:min_size], xta_test[:min_size], yt_test[:min_size]
            xm_train, ym_train = xm_train[:min_size], ym_train[:min_size]
            xm_test, ym_test = xm_test[:min_size], ym_test[:min_size]
            text_sample += yt_train.size(0)
            music_sample += ym_train.size(0)
            xt_train = {
                'input_ids': xti_train,
                'attention_mask': xta_train
            }
            xt_test = {
                'input_ids': xti_test,
                'attention_mask': xta_test
            }
            
            with torch.no_grad():
                emb_t_train = model(xt_train, type='text', return_trsfm_embedding=True)
                emb_t_test = model(xt_test, type='text', return_trsfm_embedding=True)
                emb_m_train = model(xm_train, type='music', return_trsfm_embedding=True)
                emb_m_test = model(xm_test, type='music', return_trsfm_embedding=True)
            if mode == 'gen':
                # text to music
                text_correct_train, (text_correct, text_patched_corrects), (music_zero_shot_correct, music_patched_corrects) = probing_gen(
                    x_train=emb_t_train,        # text embedding
                    y_train=yt_train,          # text label
                    x_test=emb_t_test,        # text embedding
                    y_test=yt_test,          # text label
                    Z=emb_m_test,        # music embedding
                    Yz=ym_test,          # music label
                    hidden_dim=768,
                    method_x=method_x,
                    method_z='src'
                )

                # text as train acc
                total['from_text']['text_train'] += text_correct_train
                total['from_text']['text_test'] += text_correct
                # text patched
                total['from_text']['text_patched'] = np.sum([total['from_text']['text_patched'], text_patched_corrects], axis=0)
                # zero shot generalized to music
                total['from_text']['music'] += music_zero_shot_correct
                # music patched with text
                total['from_text']['music_patched'] = np.sum([total['from_text']['music_patched'], music_patched_corrects], axis=0)
                
                
                # music to text
                music_correct_train, (music_correct, music_patched_corrects), (text_zero_shot_correct, text_patched_corrects) = probing_gen(
                    x_train=emb_m_train,        # text embedding
                    y_train=ym_train,          # text label
                    x_test=emb_m_test,        # text embedding
                    y_test=ym_test,          # text label
                    Z=emb_t_test,        # music embedding
                    Yz=yt_test,          # music label
                    hidden_dim=768,
                    method_x=method_x,
                    method_z='src'
                )


                # music as train acc
                total['from_music']['music_train'] += music_correct_train
                total['from_music']['music_test'] += music_correct
                # music patched
                total['from_music']['music_patched'] = np.sum([total['from_music']['music_patched'], music_patched_corrects], axis=0)
                # zero shot generalized to text
                total['from_music']['text'] += text_zero_shot_correct
                # text patched with music
                total['from_music']['text_patched'] = np.sum([total['from_music']['text_patched'], text_patched_corrects], axis=0)
            elif mode == 'both':
                (correct_music_train_data, correct_text_train_data), (correct_music_test_data, correct_music_patcheds), (correct_text_test_data, correct_text_patcheds) = probing(
                    music_train_data=emb_m_train,
                    music_train_label=ym_train,
                    music_test_data=emb_m_test,
                    music_test_label=ym_test,
                    text_train_data=emb_t_train,
                    text_train_label=yt_train,
                    text_test_data=emb_t_test,
                    text_test_label=yt_test,
                    hidden_dim=768,
                    method=method_x
                )
                total['text_train'] += correct_text_train_data
                total['music_train'] += correct_music_train_data
                total['text_test'] += correct_text_test_data
                total['music_test'] += correct_music_test_data
                total['text_patched'] = np.sum([total['text_patched'], correct_text_patcheds], axis=0)
                total['music_patched'] = np.sum([total['music_patched'], correct_music_patcheds], axis=0)
                
            
        assert text_sample == music_sample, "text_sample != music_sample"
        from_text_records = []
        from_music_records = []
        from_records = []
        if mode == 'gen':
            for neuron_idx in range(768):
                from_text_records.append({
                    'neuron_idx': neuron_idx,
                    'text_patched': total['from_text']['text_patched'][neuron_idx] / text_sample,
                    'music_patched': total['from_text']['music_patched'][neuron_idx] / music_sample
                })
                from_music_records.append({
                    'neuron_idx': neuron_idx,
                    'text_patched': total['from_music']['text_patched'][neuron_idx] / text_sample,
                    'music_patched': total['from_music']['music_patched'][neuron_idx] / music_sample,
                })
            from_text_df = pd.DataFrame(from_text_records)
            from_text_df['epoch'] = epoch
            from_text_df['text_train'] = total['from_text']['text_train'] / text_sample
            from_text_df['text_test'] = total['from_text']['text_test'] / text_sample
            from_text_df['music'] = total['from_text']['music'] / music_sample
            from_music_df = pd.DataFrame(from_music_records)
            from_music_df['epoch'] = epoch
            from_music_df['music_train'] = total['from_music']['music_train'] / music_sample
            from_music_df['music_test'] = total['from_music']['music_test'] / music_sample
            from_music_df['text'] = total['from_music']['text'] / text_sample
            all_from_music_df = pd.concat([all_from_music_df, from_music_df], ignore_index=True)
            all_from_text_df = pd.concat([all_from_text_df, from_text_df], ignore_index=True)
        elif mode == 'both':
            for neuron_idx in range(768):
                from_records.append({
                    'neuron_idx': neuron_idx,
                    'text_patched': total['text_patched'][neuron_idx] / text_sample,
                    'music_patched': total['music_patched'][neuron_idx] / music_sample
                })
            from_df = pd.DataFrame(from_records)
            from_df['epoch'] = epoch
            from_df['text_train'] = total['text_train'] / text_sample
            from_df['text_test'] = total['text_test'] / text_sample
            from_df['music_train'] = total['music_train'] / music_sample
            from_df['music_test'] = total['music_test'] / music_sample
            all_df = pd.concat([all_df, from_df], ignore_index=True)
            
    if mode == 'both':
        all_df.to_csv(os.path.join(result_dir, f'all_{mode}.csv'), index=False)
    else:
        all_from_text_df.to_csv(os.path.join(result_dir, f'from_text_{mode}.csv'), index=False)
        all_from_music_df.to_csv(os.path.join(result_dir, f'from_music_{mode}.csv'), index=False)
    print('probing finished')
    if mode == 'gen':
        from_text_df = pd.read_csv(os.path.join(result_dir, 'from_text.csv'))
        from_music_df = pd.read_csv(os.path.join(result_dir, 'from_music.csv'))
        result = extract_top_neurons_gen(from_music_df, from_text_df, save_dir=result_dir)
        from_music_neurons = set(result['from_music']['top_music_diff']) & set(result['from_music']['top_text_diff'])
        from_text_neurons = set(result['from_text']['top_music_diff']) & set(result['from_text']['top_text_diff'])
        print(f"Top neurons from music: {from_music_neurons}")
        print(f"Top neurons from text: {from_text_neurons}")
        neurons = list(from_music_neurons) + list(from_text_neurons)
        for neuron in neurons:
            plot_accuracy_trends(from_text_df, neuron, save_dir=result_dir)
            plot_accuracy_trends(from_music_df, neuron, save_dir=result_dir)
            plot_patch_diff_trends(from_text_df, neuron, save_dir=result_dir)
            plot_patch_diff_trends(from_music_df, neuron, save_dir=result_dir)
    elif mode == 'both':
        all_df = pd.read_csv(os.path.join(result_dir, f'all_{mode}.csv'))
        result = extract_top_neurons(all_df, save_dir=result_dir)
        neurons = set(result['top_music_diff']) & set(result['top_text_diff'])
        print(f"Top neurons: {neurons}")
        for neuron in neurons:
            plot_accuracy_trends(all_df, neuron, save_dir=result_dir)
            plot_patch_diff_trends(all_df, neuron, save_dir=result_dir)
    