import os
import random
import pretty_midi
import zipfile
from tqdm import tqdm
import string
import json
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

BASE62_ALPHABET = string.ascii_letters + string.digits

def base62_encode(num):
    if num == 0:
        return BASE62_ALPHABET[0]
    base62 = []
    while num:
        num, rem = divmod(num, 62)
        base62.append(BASE62_ALPHABET[rem])
    return ''.join(reversed(base62))

def base62_decode(encoded):
    num = 0
    for char in encoded:
        num = num * 62 + BASE62_ALPHABET.index(char)
    return num

def encode_path_base62(path):
    return base62_encode(int.from_bytes(path.encode(), 'big'))

def decode_path_base62(encoded):
    num = base62_decode(encoded)
    return num.to_bytes((num.bit_length() + 7) // 8, 'big').decode()

def extract_midi_segment_data(midi_data, start, duration):

    new_midi = pretty_midi.PrettyMIDI()
    end_time = start + duration
    for instrument in midi_data.instruments:
        new_instrument = pretty_midi.Instrument(program=instrument.program,
                                                is_drum=instrument.is_drum,
                                                name=instrument.name)
        for note in instrument.notes:
            if note.start >= start and note.end <= end_time:
                new_instrument.notes.append(
                    pretty_midi.Note(velocity=note.velocity,
                                     pitch=note.pitch,
                                     start=note.start - start,
                                     end=note.end - start)
                )
        if new_instrument.notes:
            new_midi.instruments.append(new_instrument)
    return new_midi

def extract_and_write(midi_data, start, duration, output_path):

    new_midi = extract_midi_segment_data(midi_data, start, duration)
    new_midi.write(output_path)

def check(relation, t1, d1, t2, d2, total_duration):
    if t1 + d1 > total_duration or t2 + d2 > total_duration:
        return False
    if relation == "before":
        return t1 < t1 + d1 < t2 < t2 + d2
    elif relation == "after":
        return t2 < t2 + d2 < t1 < t1 + d1
    elif relation == "meets":
        return t1 < t1 + d1 == t2 < t2 + d2
    elif relation == "overlaps" or relation == 'simultaneous':
        return t1 < t2 < t1 + d1 < t2 + d2
    elif relation == "starts":
        return t1 == t2 < t2 + d2 < t1 + d1
    elif relation == "during":
        return t1 < t2 < t2 + d2 < t1 + d1
    elif relation == "is_included":
        return t2 < t1 < t1 + d1 < t2 + d2
    elif relation == "finishes":
        return t1 < t2 < t2 + d2 == t1 + d1
    elif relation == "equals":
        return t1 == t2 and t1 + d1 == t2 + d2
    else:
        raise ValueError("Unknown relation type!")

def generate_relation_samples(midi_path, relations=["before", "after", "is_included", "simultaneous"],
                            num_pairs_per_relation=2, output_dir='data/midi_segs',
                            time_limit=30):
    path_hash = encode_path_base62(midi_path)
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    total_duration = midi_data.get_end_time()

    if total_duration < time_limit:
        raise ValueError(f"MIDI file duration {total_duration} is less than the time limit {time_limit}.")

    os.makedirs(output_dir, exist_ok=True)

    sample_pairs = {rel: [] for rel in relations}
    with ThreadPoolExecutor(max_workers=4) as write_executor:
        write_futures = []
        for rel in relations:
            rel = rel.lower()
            for pair_idx in range(num_pairs_per_relation):
                max_attempts = 100
                for attempt in range(max_attempts):
                    d1 = random.uniform(time_limit//2, min(time_limit, total_duration / 2))
                    d2 = random.uniform(time_limit//2, min(time_limit, total_duration / 2))
                    if rel not in ["before", "meets", "overlaps"]:
                        if abs(d1 - d2) < 1:
                            continue
                        d1, d2 = max(d1, d2), min(d1, d2)
                    t1 = random.uniform(0, total_duration - d2 - d1 - 1)
                    if rel == "before" or rel == "after":
                        # t2 = random.uniform(t1 + d1 + 0.5, total_duration - d2 - 0.1)
                        t2 = random.uniform(t1 + d1 + 0.5, t1 + d1*2)
                        if rel == 'after':
                            t1, t2 = t2, t1
                    elif rel == "meets":
                        t2 = t1 + d1
                    elif rel == "overlaps" or rel == 'simultaneous':
                        lower = t1 + 1
                        upper = t1 + d1 - 1
                        t2 = random.uniform(lower, upper)
                    elif rel == "starts":
                        t2 = t1
                    elif rel == "during" or rel == 'is_included':
                        lower = t1 + 1
                        upper = t1 + d1 - d2 - 1
                        t2 = random.uniform(lower, upper)
                        if rel == 'is_included':
                            t1, t2 = t2, t1
                            d1, d2 = d2, d1
                    elif rel == "finishes":
                        t2 = t1 + d1 - d2
                    elif rel == "equals":
                        d2 = d1
                        t2 = t1
                    else:
                        raise ValueError("Unknown relation type!")

                    if check(rel, t1, d1, t2, d2, total_duration):
                        break
                    else:
                        continue

                seg1_filename = "{}_pair{}_seg1_{}.mid".format(rel, pair_idx, path_hash)
                seg2_filename = "{}_pair{}_seg2_{}.mid".format(rel, pair_idx, path_hash)
                seg1_path = os.path.join(output_dir, seg1_filename)
                seg2_path = os.path.join(output_dir, seg2_filename)

                write_futures.append(write_executor.submit(extract_and_write, midi_data, t1, d1, seg1_path))
                write_futures.append(write_executor.submit(extract_and_write, midi_data, t2, d2, seg2_path))

                sample_pairs[rel].append((seg1_path, seg2_path))

        for future in as_completed(write_futures):
            future.result()

    return {path_hash: sample_pairs}

def process_midi_file(midi_file):
    try:
        samples_dict = generate_relation_samples(midi_file)
        return midi_file, samples_dict
    except Exception as e:
        return midi_file, None

if __name__ == '__main__':
    data_zip = zipfile.ZipFile('data/midi.zip', 'r')
    file_list = ['data/' + '/'.join(n.split('/')[-3:]).replace('._', '')
                 for n in data_zip.namelist()
                 if n.lower().endswith(('.mid', '.midi'))]

    print('Number of midi tracks:', len(file_list))

    seg_dict = {}
    failed = []
    # relations = ["before", "meets", "overlaps", "starts", "during", "finishes", "equals"]
    relations = ["before", "after", "is_included", "simultaneous"]
    count = {rel: 0 for rel in relations}

    num_workers = os.cpu_count() or 4
    print("Number of workers: ", num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_midi_file, midi_file): midi_file for midi_file in file_list}
        for future in tqdm(as_completed(future_to_file), total=len(file_list), desc="Processing MIDI files"):
            midi_file, samples_dict = future.result()
            if samples_dict is None:
                failed.append(midi_file)
                continue
            seg_dict.update(samples_dict)
            for pairs in samples_dict.values():
                for rel, segs in pairs.items():
                    count[rel] += len(segs)

    print('Number of samples:', count)
    print('Number of failed:', len(failed), failed)

    with open('data/midi_seg_dict.json', 'w') as f:
        json.dump(seg_dict, f, indent=4)
        