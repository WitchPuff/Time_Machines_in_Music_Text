# rm -rf data/midi_segs
# rm data/midi_seg_dict.json data/midi_oct/split_dict.json
python data/make_music_dataset.py &&
# rm -rf data/midi_oct/test
# rm -rf data/midi_oct/train
# rm -rf data/midi_oct/valid
rm data/midi_oct/split_dict.json
python data/convert_oct.py &&
# python utils/train.py 