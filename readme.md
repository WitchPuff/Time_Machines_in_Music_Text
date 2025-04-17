# Run

You can find more details and main results in `report.pdf` .



## Environment

```shell
conda env create -f environment.yml
conda activate mb
git clone https://github.com/pytorch/fairseq  
cd fairseq
git checkout 336942734c85791a90baa373c212d27e7c722662   # provided checkpoints are trained with this version
pip install --editable ./
# TIP: replace all np.float with float in fairseq for compatibility
```

## Prepare Dataset

Split text dataset is already in `data/text`.

For music dataset:

```shell
cd data

# 1. Download the file
wget http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz

# 2. Extract the tar.gz file
tar -xzf clean_midi.tar.gz

# 3. Change the formats to fit in the shell scripts
zip -r midi.zip clean_midi/

sh make_music_dataset.sh
```

The final document structure of `data` document would look like this:

```shell
❯ tree -L 1
.
├── __init__.py
├── ckpt # pretrained checkpoints for MusicBERT and RoBERTa
├── clean_midi # Unzip midi.zip
├── convert_oct.py
├── dataset_overview.py
├── helper.py
├── make_lang_dataset.py
├── make_music_dataset.py
├── make_music_dataset.sh
├── midi.zip
├── midi_oct # Already Encoded with OctupleMIDI
├── midi_seg_dict.json
├── midi_segs # sampled MIDI segments
└── text

```

`data/dataset_overview.py` will print the detailed statistics of datasets.

```shell
❯ python "data/dataset_overview.py"

===== Dataset overview =====

===== Music dataset overview =====

train 108608
valid 13576
test 13576
===== Text dataset overview =====

File: test_data.csv
Answer
BEFORE          3814
AFTER           3431
IS_INCLUDED     1583
SIMULTANEOUS     924
Name: count, dtype: int64
9752

File: train_data.csv
Answer
BEFORE          30825
AFTER           27162
IS_INCLUDED     12474
SIMULTANEOUS     7557
Name: count, dtype: int64
78018

File: valid_data.csv
Answer
BEFORE          3829
AFTER           3416
IS_INCLUDED     1556
SIMULTANEOUS     952
Name: count, dtype: int64
9753
```



## Train

Prepare pretrained checkpoints for RoBERTa and MusicBERT:

- checkpoint_last_musicbert_base_w_genre_head.pt 

  ```shell
  gdown --id 1bN1DEFwCz9b3du13Ai9SezujzUlzeRSJ
  mv checkpoint_last_musicbert_base_w_genre_head.pt data/ckpt/
  ```

- Clone the checkpoints of RoBERTa from [Hugging Face](https://huggingface.co/FacebookAI/roberta-base).

The `data/ckpt` is structured as:

```shell
❯ tree
.
├── checkpoint_last_musicbert_base_w_genre_head.pt
└── roberta-base
    ├── config.json
    ├── dict.txt
    ├── merges.txt
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```

Then run the train script:

```shell
python utils/train.py \
    --epochs 60 \
    --batch_size 32 \
    --text_max_length 512 \
    --music_max_length 1024 \
    --sample_size 78016 \
    --warmup_step 10000 \
    --decay_step 100000 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --ckpt_dir train_logs/ckpt \
    --branch music_text
```

The checkpoints during training process will be stored in `train_logs/ckpt`. The best checkpoints we use for further analysis are still being uploaded for they are large files.

## Analysis

```shell
# perform attention analysis and visualize the trends for music data per temporal relation label
# All orginial table data in csv from, as well as plotted figures are stored in `music_attn`
python music_attn.py 

# perform attention analysis and visualize the trends for text data per temporal relation label
# All orginial table data in csv from, as well as plotted figures are stored in `text_attn`
python text_attn.py

# activation patching and probing analysis for both music and text data
# All orginial table data in csv from, as well as plotted figures are stored in `probing`
python probing.py
```

 

