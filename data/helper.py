import json

with open('data/midi_oct/split_dict.json', 'r') as f:
    split_dict = json.load(f)

for set, keys in split_dict.items():
    print(set, len(keys)*4)