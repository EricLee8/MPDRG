import os
import json
import torch
import numpy as np
import random
import warnings
import importlib
from tqdm import tqdm
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizerFast
from transformers import BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from pycocoevalcap.eval import COCOEvalCap
from utils.config import *
from utils.utils import text, get_dataset, collate_fn, read_examples


GenerationModel = importlib.import_module('models.' + args.model_file).GenerationModel


MODEL_CLASSES = {
    'bart': (BartConfig, BartTokenizerFast)
}


warnings.filterwarnings("ignore")
device = torch.device("cuda:"+str(args.cuda)) if USE_CUDA else torch.device("cpu")
train_path = os.path.join(args.data_path, args.dataset, "train.json")
eval_path = os.path.join(args.data_path, args.dataset, "valid.json")
test_path = os.path.join(args.data_path, args.dataset, "test.json")
config_class, tokenizer_class = MODEL_CLASSES[args.model_type]


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def cur_larger(cur_result, cur_best_result):
    for metric in METRICS:
        if cur_result[metric] != cur_best_result[metric]:
            return cur_result[metric] > cur_best_result[metric]
    return cur_result[metric[-1]] > cur_best_result[metric[-1]]


def prepare_inputs(batch, training=True, estep=False):
    if training:
        inputs = {'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'decoder_attention_mask': batch['decoder_attention_mask'],
                    'labels': batch['labels']
                    }
    else:
        inputs = {'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'training': False
                    }
    if args.indicator:
        inputs.update({'indicator_ids': batch['indicator_ids']})
    if estep:
        inputs.update({'estep': True})
    
    return inputs 


def train(model, train_loader, eval_dataloader, test_dataloader, tokenizer):
    print("Traning arguments:")
    print(args)
    
    best_result = {metric: 0.0 for metric in METRICS}
    model.train()
    model.zero_grad()

    if args.fp16:
        scaler = GradScaler()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    logging_step = len(train_loader)
    steps = 0

    # eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name='eval_result.json')
    # evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, file_name='test_result.json')

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        for _, batch in pbar:
            inputs = prepare_inputs(batch)

            if args.fp16:
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs['loss']
                    print_loss = loss.item()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.unscale_(all_optimizer)
                    scaler.step(all_optimizer)
                    scaler.update()
            else:
                outputs = model(**inputs)
                loss = outputs['loss']
                print_loss = loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                all_optimizer.step()

            scheduler.step()
            model.zero_grad()

            pbar.set_description("Loss:%.3f,CL:%.3f" %(print_loss, print_loss))
            if steps != 0 and steps % logging_step == 0 or steps == t_total-1:
                print("\nEpoch {}, Step {}".format(epoch, steps))
                eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name='eval_result.json')
                if eval_result['ROUGE_L'] < 0.1:
                    print("Become untrainable! Training canceled!!!")
                    return
                if cur_larger(eval_result, best_result):
                    best_result = eval_result
                evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, file_name='test_result.json')
            steps += 1


def evaluate(model, eval_loader, tokenizer, cur_best_result=None, is_test=False, file_name=None, cal_metrics=True):
    def _cal_metrics(golden_dict, pred_dict):
        golden_dict = {qid: golden_dict[qid] for qid in pred_dict.keys()}
        evaluator = COCOEvalCap(text(list(golden_dict.values())), text(list(pred_dict.values())))
        evaluator.evaluate()
        result_dict = {metric: score*100 for metric, score in evaluator.eval.items()}
        return result_dict

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100)
        pred_dict = {}

        for _, batch in pbar:
            inputs = prepare_inputs(batch, training=False)

            summary_ids_list = model(**inputs)
            for summary_ids, qid in zip(summary_ids_list, batch['qid']):
                decoded_summary = tokenizer.decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pred_dict[qid] = decoded_summary
    
    if cal_metrics:
        examples = read_examples(test_path if is_test else eval_path, training=False)
        golden_dict = {exp.qid: exp.response_info['response'] for exp in examples}
        result_dict = _cal_metrics(golden_dict, pred_dict)
        print("Test Result:" if is_test else "Eval Result:", result_dict)
        if cur_best_result is not None:
            if cur_larger(result_dict, cur_best_result):
                print("model and arguments saved to {}...".format(args.save_path))
                save_path = os.path.join(args.save_path, "best_model.pth")
                args_save_path = os.path.join(args.save_path, "args.pth")
                model = model.to("cpu")
                torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
                model = model.to(device)
                torch.save(args, args_save_path, _use_new_zipfile_serialization=False)

    model.train()

    if file_name is not None:
        with open(os.path.join(args.save_path, file_name), "w", encoding='utf-8') as f:
            json.dump(pred_dict, f, indent=2)

    return result_dict if cal_metrics else None


def training(model, tokenizer):
    train_dataset = get_dataset(train_path, args.cache_path,\
            tokenizer, training=True)
    eval_dataset = get_dataset(eval_path, args.cache_path,\
            tokenizer, training=False)
    test_dataset = get_dataset(test_path, args.cache_path,\
            tokenizer, training=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    if hasattr(model.model, 'load_mha_params'):
        print("Loading multi-head attention parameters from pretrained model...")
        model.model.load_mha_params()
    if args.pretrain_path is not None:
        print("Loading pre-trained model from {}...".format(args.pretrain_path))
        model_saved = torch.load(args.pretrain_path)
        if args.del_indicator_embs:
            print("Deleting weights of indicator_embs...")
            del model_saved['indicator_embs.weight']
        model.load_state_dict(model_saved, strict=not args.del_indicator_embs)
    model = model.to(device)
    train(model, train_dataloader, eval_dataloader, test_dataloader, tokenizer)


if __name__ == "__main__":
    set_seed()
    
    print("Loading tokenizer and model...")
    tokenizer = tokenizer_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
    config = config_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
    model = GenerationModel.from_pretrained(args.model_name, config=config, cache_dir=args.cache_path)
    tokenizer.truncation_side = 'left'

    training(model, tokenizer)