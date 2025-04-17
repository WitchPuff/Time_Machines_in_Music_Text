# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import json
import sys
from tqdm import tqdm
import io
import zipfile
import miditoolkit
import random
import time
import math
import signal
import hashlib
from multiprocessing import Pool, Lock, Manager
import multiprocessing
multiprocessing.set_start_method("fork")

pos_resolution = 16  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 8  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 1000  # window length max
sample_overlap_rate = 4
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127
max_velocity = 127

data_zip = ''
output_file = ''

lock_file = Lock()
lock_write = Lock()
lock_set = Lock()
midi_dict = {}  # thread-safe dict





# (0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo)
# (Measure, TimeSig)
# (Pos, Tempo)
# Percussion: Program=128 Pitch=[128,255]


ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)



class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)


def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def e2t(x):
    return ts_list[x]


def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]


def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]


def v2e(x):
    return x // velocity_quant


def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)


def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def e2b(x):
    return 2 ** (x / tempo_quant) * min_tempo


def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator


def writer(file_name, output_str_list):
    # note: parameter "file_name" is reserved for patching
    global output_file
    with open(output_file, 'a') as f:
        for output_str in output_str_list:
            f.write(output_str + '\n')


def gen_dictionary(file_name):
    num = 0
    with open(file_name, 'w') as f:
        for j in range(bar_max):
            print('<0-{}>'.format(j), num, file=f)
        for j in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            print('<1-{}>'.format(j), num, file=f)
        for j in range(max_inst + 1 + 1):
            # max_inst + 1 for percussion
            print('<2-{}>'.format(j), num, file=f)
        for j in range(2 * max_pitch + 1 + 1):
            # max_pitch + 1 ~ 2 * max_pitch + 1 for percussion
            print('<3-{}>'.format(j), num, file=f)
        for j in range(duration_max * pos_resolution):
            print('<4-{}>'.format(j), num, file=f)
        for j in range(v2e(max_velocity) + 1):
            print('<5-{}>'.format(j), num, file=f)
        for j in range(len(ts_list)):
            print('<6-{}>'.format(j), num, file=f)
        for j in range(b2e(max_tempo) + 1):
            print('<7-{}>'.format(j), num, file=f)


def MIDI_to_encoding(midi_obj):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = min(max(notes_start_pos) + 1, trunc_pos)
    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
    start_distribution = [0] * pos_resolution
    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]
            encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                             1 if inst.is_drum else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), v2e(note.velocity), info[1], info[3]))
    if len(encoding) == 0:
        return list()
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                          math.log2((x / tot)) for x in start_distribution))
    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)
    encoding.sort()
    return encoding


def encoding_to_MIDI(encoding):
    # TODO: filter out non-valid notes and error handling
    bar_to_timesig = [list()
                      for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = e2t(bar_to_timesig[i])
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = (i[3] - 128 if program == 128 else i[3])
        duration = get_tick(0, e2d(i[4]))
        if duration == 0:
            duration = 1
        end = start + duration
        velocity = e2v(i[5])
        midi_obj.instruments[program].notes.append(miditoolkit.containers.Note(
            start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj


def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode('ascii')).hexdigest()
    return midi_hash


def sample_middle_preserve_ends(events, max_tokens=8192, num_preserve_front=100, num_preserve_end=100):
    """
    保留 events 的前 num_preserve_front 和后 num_preserve_end 个 token，
    对中间部分均匀采样，使得最终长度不超过 max_tokens。
    """
    if len(events) <= max_tokens:
        return events

    # 保留开头和结尾部分
    front = events[:num_preserve_front]
    back = events[-num_preserve_end:]
    middle = events[num_preserve_front: len(events) - num_preserve_end]

    # 中间部分需要采样到的目标长度
    target_middle_len = max_tokens - (num_preserve_front + num_preserve_end)
    if target_middle_len <= 0:
        # 如果头尾已经超过最大长度，则只保留头尾（或根据需求处理）
        return front + back

    # 均匀采样中间部分：计算采样步长
    step = len(middle) / target_middle_len
    sampled_middle = [middle[int(i * step)] for i in range(target_middle_len)]

    return front + sampled_middle + back

def F(file_name):
    global midi_dict, output_file, lock_file, lock_write, lock_set
    try_times = 10
    midi_file = None
    for _ in range(try_times):
        try:
            global lock_file
            lock_file.acquire()
            with open(file_name, 'rb') as f:
                # this may fail due to unknown bug
                midi_file = io.BytesIO(f.read())
        except BaseException as e:
            print(e)
            try_times -= 1
            time.sleep(1)
            if try_times == 0:
                print('ERROR(READ): ' + file_name +
                      ' ' + str(e) + '\n', end='')
                return None
        finally:
            lock_file.release()
    try:
        with timeout(seconds=600):
            midi_obj = miditoolkit.midi.parser.MidiFile(file=midi_file)
        # check abnormal values in parse result
        assert all(0 <= j.start < 2 ** 31 and 0 <= j.end < 2 **
                   31 for i in midi_obj.instruments for j in i.notes), 'bad note time'
        assert all(0 < j.numerator < 2 ** 31 and 0 < j.denominator < 2 **
                   31 for j in midi_obj.time_signature_changes), 'bad time signature value'
        assert 0 < midi_obj.ticks_per_beat < 2 ** 31, 'bad ticks per beat'
    except BaseException as e:
        print('ERROR(PARSE): ' + file_name + ' ' + str(e) + '\n', end='')
        return None
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    try:
        e = MIDI_to_encoding(midi_obj)
        if len(e) == 0:
            print('ERROR(BLANK): ' + file_name + '\n', end='')
            return None
        if ts_filter:
            allowed_ts = t2e(time_signature_reduce(4, 4))
            if not all(i[6] == allowed_ts for i in e):
                print('ERROR(TSFILT): ' + file_name + '\n', end='')
                return None
        if deduplicate:
            duplicated = False
            dup_file_name = ''
            midi_hash = '0' * 32
            try:
                midi_hash = get_hash(e)
            except BaseException as e:
                pass
            lock_set.acquire()
            if midi_hash in midi_dict:
                dup_file_name = midi_dict[midi_hash]
                duplicated = True
            else:
                midi_dict[midi_hash] = file_name
            lock_set.release()
            if duplicated:
                print('ERROR(DUPLICATED): ' + midi_hash + ' ' +
                      file_name + ' == ' + dup_file_name + '\n', end='')
                return None
        output_str_list = []
        e_segment = sample_middle_preserve_ends(e)
        # sample_step = max(round(sample_len_max / sample_overlap_rate), 1)
        # for p in range(0 - random.randint(0, sample_len_max - 1), len(e), sample_step):
        #     L = max(p, 0)
        #     R = min(p + sample_len_max, len(e)) - 1
        #     bar_index_list = [e[i][0]
        #                       for i in range(L, R + 1) if e[i][0] is not None]
        #     bar_index_min = 0
        #     bar_index_max = 0
        #     if len(bar_index_list) > 0:
        #         bar_index_min = min(bar_index_list)
        #         bar_index_max = max(bar_index_list)
        #     offset_lower_bound = -bar_index_min
        #     offset_upper_bound = bar_max - 1 - bar_index_max
        #     # to make bar index distribute in [0, bar_max)
        #     bar_index_offset = random.randint(
        #         offset_lower_bound, offset_upper_bound) if offset_lower_bound <= offset_upper_bound else offset_lower_bound
        #     e_segment = []
        #     for i in e[L: R + 1]:
        #         if i[0] is None or i[0] + bar_index_offset < bar_max:
        #             e_segment.append(i)
        #         else:
        #             break
        tokens_per_note = 8
        output_words = encoding_to_str(e_segment)
        # output_words = (['<s>'] * tokens_per_note) \
        #     + [('<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) if k is not None else '<unk>') for i in e_segment for j, k in enumerate(i)] \
        #     + (['</s>'] * (tokens_per_note - 1)
        #         )  # tokens_per_note - 1 for append_eos functionality of binarizer in fairseq
        output_str_list.append(output_words)
        assert (len(output_str_list) == 1) and (len(e_segment) <= 8192)
        # no empty
        if not all(len(i.split()) > tokens_per_note * 2 - 1 for i in output_str_list):
            print('ERROR(ENCODE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        try:
            lock_write.acquire()
            writer(file_name, output_str_list)
        except BaseException as e:
            print('ERROR(WRITE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        finally:
            lock_write.release()
        # print('SUCCESS: ' + file_name + '\n', end='')
        return True
    except BaseException as e:
        print('ERROR(PROCESS): ' + file_name + ' ' + str(e) + '\n', end='')
        return False
    print('ERROR(GENERAL): ' + file_name + '\n', end='')
    return False


def G(file_name):
    try:
        return F(file_name)
    except BaseException as e:
        print('ERROR(UNCAUGHT): '+ str(e) + file_name + '\n', end='')
        return False


def str_to_encoding(s):
    encoding = [int(i[3: -1]) for i in s.split() if 's' not in i]
    tokens_per_note = 8
    assert len(encoding) % tokens_per_note == 0
    encoding = [tuple(encoding[i + j] for j in range(tokens_per_note))
                for i in range(0, len(encoding), tokens_per_note)]
    return encoding


def encoding_to_str(e):
    bar_index_offset = 0
    p = 0
    tokens_per_note = 8
    return ' '.join((['<s>'] * tokens_per_note)
                    + ['<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) for i in e[p: p +
                                                                                            sample_len_max] if i[0] + bar_index_offset < bar_max for j, k in enumerate(i)]
                    + (['</s>'] * (tokens_per_note
                                   - 1)))   # 8 - 1 for append_eos functionality of binarizer in fairseq
# def initialize_global():
#     global manager, midi_dict, output_file, data_zip
#     try:
#         # Initialize the manager and midi_dict
#         manager = Manager()
#         midi_dict = manager.dict()  # thread-safe dict
#         data_path = input('Dataset path: ')
#         data_zip = zipfile.ZipFile(data_path, 'r')
#         print("Manager, midi_dict, data_zip initialized")
#     except Exception as e:
#         print(f"ERROR initializing global variables: {e}")

# if __name__ == '__main__':
    # initialize_global()
    # # Check if midi_dict is initialized correctly
    # if midi_dict is None:
    #     print("midi_dict is NOT initialized correctly.")
    # else:
    #     print("midi_dict initialized successfully.")
    

# data_path = 'data/midi.zip'
# data_zip = zipfile.ZipFile(data_path, 'r')
# if os.path.exists(prefix):
#     print('Output path {} already exists!'.format(prefix))
#     sys.exit(0)
# file_list = ['data/'+'/'.join(n.split('/')[-3:]).replace('._', '') for n in data_zip.namelist() if n[-4:].lower()
#                 == '.mid' or n[-5:].lower() == '.midi']
# random.shuffle(file_list)
prefix = 'data/midi_oct'
os.makedirs(prefix, exist_ok=True)
gen_dictionary('{}/dict.txt'.format(prefix))
with open('data/midi_seg_dict.json', 'r') as f:
    data_dict = json.load(f)
midi_keys = list(data_dict.keys())
random.shuffle(midi_keys)

split_dict = {}
ok_cnt = 0
all_cnt = 0
relations = ["before", "after", "is_included", "simultaneous"]

if os.path.exists('data/midi_oct/split_dict.json'):
    with open('data/midi_oct/split_dict.json', 'r') as f:
        split_dict = json.load(f)
for sp in ['train', 'valid', 'test']:
    total_track_cnt = len(midi_keys)
    print(total_track_cnt)
    if sp in split_dict: midi_keys_split = split_dict[sp]
    else: 
        print('Randomly split dataset...')
        if sp == 'train':  # 80%
            midi_keys_split = midi_keys[: 80 * total_track_cnt // 100]
        if sp == 'valid':  # 10%
            midi_keys_split = midi_keys[80 * total_track_cnt //
                                        100: 90 * total_track_cnt // 100]
        if sp == 'test':  # 10%
            midi_keys_split = midi_keys[90 * total_track_cnt // 100:]
        split_dict[sp] = midi_keys_split
    for rel in relations:
        if rel == "equals": deduplicate = False
        else: deduplicate = True
        os.makedirs(os.path.join(prefix, sp, rel), exist_ok=True)
        midi_list = []
        for key in tqdm(midi_keys_split):
            for pi, pair in enumerate(data_dict[key][rel]):
                output_file = os.path.join(prefix, sp, rel, f'{key}_{pi}.txt')
                for seg in pair:
                    result = G(seg)
                    all_cnt += 1
                    if not result:
                        if os.path.exists(output_file): os.remove(output_file)
                        break
                    else: ok_cnt += 1
                # with Pool(pool_num) as p:
                #     result = list(p.imap_unordered(G, pair))
                #     all_cnt += sum((1 if i is not None else 0 for i in result))
                #     ok_cnt += sum((1 if i is True else 0 for i in result))
                output_file = ''
if not os.path.exists('data/midi_oct/split_dict.json'):
    with open('data/midi_oct/split_dict.json', 'w') as f:
        json.dump(split_dict, f, indent=4)    
print('{}/{} ({:.2f}%) MIDI files successfully processed'.format(ok_cnt,
                                                                    all_cnt, ok_cnt / all_cnt * 100))