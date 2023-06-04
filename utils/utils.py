import os
import torch
import json
import multiprocessing
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from .config import *


class text(object):
    def __init__(self, lines):
        self.imgToAnns = {}
        for _id, line in enumerate(lines):
            self.imgToAnns[_id] = [{"caption": line}]
        
    def getImgIds(self):
        return self.imgToAnns.keys()


class Example(object):
    def __init__(self, utterances, speakers, qid, response_info=None):
        self.utterances = utterances
        self.speakers = speakers
        self.qid = qid
        self.response_info = response_info

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "utterances: " + self.utterances + '\n'
        s += "speakers: " + self.speakers + '\n'
        s += "qid: " + self.qid + '\n'
        s += "response_info: "  + self.response_info
        return s


class InputFeature(object):
    def __init__(self, qid, input_ids, attention_mask, sep_poses,\
         indicator_ids, decoder_attention_mask=None, labels=None):
        self.qid = qid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sep_poses = sep_poses
        self.indicator_ids = indicator_ids
        self.decoder_attention_mask = decoder_attention_mask
        self.labels = labels


class Dataset(data.Dataset):
    def __init__(self, features):
        self.features = features
    
    def __getitem__(self, index):
        data_info = {}
        data_info['qid'] = self.features[index].qid
        data_info['input_ids'] = torch.tensor(self.features[index].input_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(self.features[index].attention_mask, dtype=torch.long)
        data_info['sep_poses'] = self.features[index].sep_poses
        data_info['indicator_ids'] = torch.tensor(self.features[index].indicator_ids, dtype=torch.long) if\
            self.features[index].indicator_ids is not None else None
        data_info['decoder_attention_mask'] = torch.tensor(self.features[index].decoder_attention_mask, dtype=torch.long) if\
            self.features[index].decoder_attention_mask is not None else None
        data_info['labels'] = torch.tensor(self.features[index].labels, dtype=torch.long) if\
            self.features[index].labels is not None else None
        return data_info
    
    def __len__(self):
        return len(self.features)


def _cuda(x):
    if USE_CUDA:
        return x.cuda(device="cuda:"+str(args.cuda))
    else:
        return x


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def filter_simple(utterances, response, qid, cache_dict=None):
    if cache_dict is not None:
        return (cache_dict[qid]['has_key_word'] or max(cache_dict[qid]['rl']) > args.filter_simple,\
             cache_dict[qid]['has_key_word'], cache_dict[qid]['rl'])
    for kw in FILTER_KEY_WORDS:
        if kw in response:
            return (True, True, [1.0])
    rl = []
    for utter in utterances:
        rl.append(EVALUATOR.get_scores(utter, response)['rouge-l']['f'])
    return (max(rl) > args.filter_simple, False, rl)


def collate_fn(data):
    data_info = {}
    float_type_keys = []
    for k in data[0].keys():
        data_info[k] = [d[k] for d in data]
    for k in data_info.keys():
        if isinstance(data_info[k][0], torch.Tensor):
            data_info[k] = _cuda(torch.stack(data_info[k]))
        if isinstance(data_info[k][0], dict):
            new_dict = {}
            for id_key in data_info[k][0].keys():
                if data_info[k][0][id_key] is None:
                    new_dict[id_key] = None
                    continue
                id_key_list = [torch.tensor(sub_dict[id_key], dtype=torch.long if id_key not in float_type_keys else torch.float) for sub_dict in data_info[k]] # (bsz, seqlen)
                id_key_tensor = torch.stack(id_key_list)
                new_dict[id_key] = _cuda(id_key_tensor)
            data_info[k] = new_dict
    return data_info


def read_examples(input_file, training=True):
    prefix = ""
    for _type in ['train', 'valid', 'test', 'case']:
        if _type in input_file:
            prefix = _type
            break

    examples = []
    max_utter_num, filter_num = 0, 0
    print("Reading examples from {}...".format(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        lines = reader.readlines()
    
    if training and args.filter_simple > 0:
        rouge_cache_file = input_file.replace(prefix, prefix+"-rouge")
        if os.path.exists(rouge_cache_file):
            with open(rouge_cache_file, "r", encoding='utf-8') as f:
                cache_dict = json.load(f)
        else:
            new_cache_dict = {}
            cache_dict = None

    for idx, line in enumerate(tqdm(lines, ncols=100)):
        qid = prefix + '-' + str(idx)
        record = json.loads(line.strip())
        utterances = record['context']
        speakers = ["#speaker{}#".format(x) for x in record['ctx_spk']]
        # addressee_ids = [x-1 if x>0 else x for x in record['ctx_adr']]
        response_info = {}
        response_info['response'] = record['answer']
        response_info['ans_spk'] = "#speaker{}#".format(record['ans_spk'])
        response_info['ans_adr'] = record['ans_adr'] - 1
        response_info['ans_idx'] = record['ans_idx'] - 1
        if response_info['ans_adr'] < 0 or response_info['ans_adr'] >= len(utterances):
            response_info['ans_adr'] = len(utterances) - 1

        max_utter_num = max(max_utter_num, len(utterances))
        exp = Example(utterances, speakers, qid, response_info)
        if training and args.filter_simple > 0:
            is_simple, has_key, rl = filter_simple(exp.utterances, exp.response_info['response'], exp.qid, cache_dict)
            if cache_dict is None:
                new_cache_dict[qid] = {"has_key_word": has_key, 'rl': rl}
            if is_simple:
                filter_num += 1
                continue
        examples.append(exp)

    if training and args.filter_simple > 0 and cache_dict is None:
        with open(rouge_cache_file, "w", encoding='utf-8') as f:
            json.dump(new_cache_dict, f, indent=2)

    print("Max utterance num: {}".format(max_utter_num))
    if training and args.filter_simple > 0:
        print("Filter threshold: {}, filtered ratio: {}%".format(round(args.filter_simple, 3), round(filter_num/len(lines) * 100, 3)))
    if args.small:
        examples = examples[:100] if training else examples[:100]
    return examples


def read_examples_estep(input_file, training=True):
    pass


def pretrain_read_examples_estep(input_file, training=True):
    pass


def pretrain_read_examples_mstep(input_file, training=True):
    pass


def convert_single_example(item):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    def _get_sep_poses(input_ids):
        sep_poses = []
        last_idx = 0 # except the [CLS]
        for idx, inpidx in enumerate(input_ids):
            if inpidx == tokenizer.sep_token_id:
                sep_poses.append((last_idx+1, idx+1))
                last_idx = idx
        return sep_poses

    exp, tokenizer, training, return_statistics = item

    context = ''
    response_info = exp.response_info
    for speaker, utterance in list(zip(exp.speakers, exp.utterances))[:len(exp.utterances) if args.addr not in [2] else response_info['ans_adr']+1]:
        context += tokenizer.sep_token + ' ' + speaker + ": " + utterance + ' '
    context = context.strip()[len(tokenizer.sep_token)+1:] # remove the first sep token and ' '
    if args.addr in [0, 2]:
        context += ' ' + tokenizer.sep_token + ' ' + response_info['ans_spk'] + ":" # only use the speaker information and leave the addressee information
    elif args.addr == 1:
        context += ' ' + tokenizer.sep_token + ' respond to [{}].'.format(exp.utterances[response_info['ans_adr']])
        context += ' ' + tokenizer.sep_token + ' ' + response_info['ans_spk'] + ":"

    ids_dict = tokenizer.encode_plus(context, padding='max_length',\
            truncation=True, max_length=args.max_length)
    input_ids = ids_dict['input_ids']
    attention_mask = ids_dict['attention_mask']

    if return_statistics:
        text_len = len(tokenizer.encode(context))
        too_long = text_len > args.max_length

    sep_poses = _get_sep_poses(input_ids)
    indicator_ids = [0] * len(input_ids)
    offset = len(exp.utterances) - (len(sep_poses) - (2 if args.addr == 1 else 1)) # deal with truncation
    true_addr = response_info['ans_adr'] - offset
    if true_addr >= 0:
        s, e = sep_poses[true_addr]
        for idx in range(s, e):
            indicator_ids[idx] = 1
    if args.addr == 1:
        s, e = sep_poses[-2]
        for idx in range(s, e):
            indicator_ids[idx] = 1

    # inference
    if not training:
        f_tmp = InputFeature(exp.qid, input_ids, attention_mask, sep_poses, indicator_ids)
        if return_statistics:
            return f_tmp, text_len, too_long
        else:
            return f_tmp
    # training
    response = response_info['response']

    if return_statistics:
        response_len = len(tokenizer.encode(response))
        response_too_long = response_len > args.response_max_length

    decoder_ids_dict = tokenizer.encode_plus(response, padding='max_length', truncation=True, max_length=args.response_max_length)
    labels = decoder_ids_dict['input_ids']
    decoder_attention_mask = decoder_ids_dict['attention_mask']
    f_tmp = InputFeature(exp.qid, input_ids, attention_mask, sep_poses, indicator_ids, decoder_attention_mask, labels)
    if return_statistics:
        return f_tmp, text_len, too_long, response_len, response_too_long
    else:
        return f_tmp


def convert_examples_to_features(examples, tokenizer, training=True):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    print("Converting examples to features...")
    max_tokens, response_max_tokens = 0, 0
    total_num, too_long_num, response_too_long_num = len(examples), 0, 0

    features = []
    for exp in tqdm(examples, ncols=100):
        # inference
        if not training:
            f_tmp, text_len, too_long = convert_single_example((exp, tokenizer, training, True))
            max_tokens = max(max_tokens, text_len)
            if too_long: too_long_num += 1
            features.append(f_tmp)
            continue

        # training
        f_tmp, text_len, too_long, response_len, response_too_long = convert_single_example((exp, tokenizer, training, True))
        max_tokens = max(max_tokens, text_len)
        if too_long: too_long_num += 1
        response_max_tokens = max(response_max_tokens, response_len)
        if response_too_long: response_too_long_num += 1
        features.append(f_tmp)

    print("Max token length:", max_tokens)
    print("Truncation num: %d, truncation rate %.2f%%" %(too_long_num, too_long_num/total_num * 100))
    if training:
        print("Max response token length:", response_max_tokens)
        print("response truncation num: %d, truncation rate %.2f%%" %(response_too_long_num, response_too_long_num/total_num * 100))

    return features


def get_dataset(input_file, save_path, tokenizer, training=True, e_step=False):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    postfix = ""
    for type_ in ["train", "valid", "test"]:
        if type_ in input_file:
            postfix = type_
            break

    examples = read_examples_estep(input_file, training=training) if e_step else read_examples(input_file, training=training)
    
    print("Converting examples to features...")
    parameters = [(exp, tokenizer, training, False) for exp in examples]
    pool = multiprocessing.Pool(CPU_COUNT)
    features = pool.map(convert_single_example, tqdm(parameters, ncols=100, total=len(parameters)))

    dataset = Dataset(features)
    return dataset


def get_pretrain_dataset(input_file, save_path, tokenizer, training=True, e_step=False):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    postfix = ""
    for type_ in ["train", "valid", "test"]:
        if type_ in input_file:
            postfix = type_
            break

    examples = pretrain_read_examples_estep(input_file, training=training) if e_step\
         else pretrain_read_examples_mstep(input_file, training=training)

    print("Converting examples to features...")
    parameters = [(exp, tokenizer, training, False) for exp in examples]
    pool = multiprocessing.Pool(CPU_COUNT)
    features = pool.map(convert_single_example, tqdm(parameters, ncols=100, total=len(parameters)))

    dataset = Dataset(features)
    return dataset

    
if __name__ == "__main__":
    input_file = "data/{}/test.json".format(args.dataset)

    from transformers import BartTokenizerFast
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    tokenizer.truncation_side = 'left'
    dataset = get_dataset(input_file, "tmp", tokenizer, training=True)# if 'test' not in input_file else False)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    for batch in tqdm(dataloader, ncols=100):
        print(batch["qid"][0])
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
        print(batch["labels"].shape)
        print(batch["sep_poses"][0])
