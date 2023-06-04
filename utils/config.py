import os
import argparse
from rouge import Rouge

USE_CUDA = True
CPU_COUNT = 16
METRICS = ['Bleu_4', 'ROUGE_L', 'METEOR', 'Bleu_1', 'Bleu_2', 'Bleu_3']
FILTER_KEY_WORDS = ["do n't know", "does n't know", "i 'm not sure", "'ll give it a"]
EVALUATOR = Rouge(metrics=['rouge-n', 'rouge-l'],
                    max_n=2,
                    limit_length=True,
                    length_limit=128,
                    length_limit_type='words',
                    apply_avg=True,
                    apply_best=False,
                    alpha=0.5, # Default F1_score
                    weight_factor=1.2,
                    stemming=True)

parser = argparse.ArgumentParser(description='Parameters for Ubuntu IRC Dataset')

parser.add_argument('--store_path', type=str, default='.', help="path for storage (output, checkpoints, caches, data, etc.)")

parser.add_argument('-lr', '--learning_rate', type=float, default=4e-5)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1919810)
parser.add_argument('-eps', '--epochs', type=int, default=15)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-mt', '--model_type', type=str, default='bart')
parser.add_argument('-cp', '--cache_path', type=str, default='cache')
parser.add_argument('-ml', '--max_length', type=int, default=384)
parser.add_argument('-sml', '--response_max_length', type=int, default=50)
parser.add_argument('-bsz', '--batch_size', type=int, default=64)
parser.add_argument('-dbg', '--debug', type=bool, default=False)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--small', type=bool, default=False, help='whether to use small dataset')
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--dataset', type=str, default='ubuntu')

# parser.add_argument('--model_name', type=str, default='lidiya/bart-large-xsum-samsum')
parser.add_argument('--model_name', type=str, default='facebook/bart-base')
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--model_file', type=str, default='baseline')
parser.add_argument('--fp16', type=int, default=1)

parser.add_argument('--addr', type=int, default=0, help='mode to use the addressee information')
# 0: no addressee information
# 1: concatenate the addressee information to the end of the dialogue context (like a prompt)
# 2: discard the utterances after the addressee, which means the last utterance is always spoken by the addressee

parser.add_argument('--indicator', type=int, default=0, help='whether to use indicator embeddings to indicate addressees')
parser.add_argument('--filter_simple', type=float, default=-1.0, help='filter simple responses which can be extract from the context, if rouge-l > this value, filtered')
parser.add_argument('--pretrain_path', type=str, default=None, help='whether to use EM pretrained model to finetune, if not None then yes')
parser.add_argument('--del_indicator_embs', type=int, default=0, help='whether to delete indicator embedding weights during finetuning')

args = parser.parse_args()

args.data_path = os.path.join(args.store_path, args.data_path)

if "large" in args.model_name:
    args.learning_rate = args.learning_rate / 4
    args.batch_size = args.batch_size // 4
    args.save_path = 'lgsave'
if args.model_file == 'embindicator':
    args.indicator = 1
if args.pretrain_path is not None:
    args.save_path = "finesave"
    if args.del_indicator_embs:
        args.save_path = "delfinesave"

save_root = '{}_saves'.format(args.dataset) if not args.small else '{}_saves_small'.format(args.dataset)
cache_root = 'caches' if not args.small else 'caches_samll'
save_root = os.path.join(args.store_path, save_root)
cache_root = os.path.join(args.store_path, cache_root)

if not os.path.exists(save_root):
    os.mkdir(save_root)
if not os.path.exists(cache_root):
    os.mkdir(cache_root)

args.save_path = save_root + '/' + args.model_type + '_addrmode{}'.format(args.addr) +\
     '_' + '{}lr{}'.format(args.model_file, args.learning_rate) + '_filter{}_'.format(args.filter_simple) + args.save_path
args.cache_path = cache_root + '/' + args.model_type + '_' + args.cache_path

args.cache_path += '_' + str(args.max_length)
args.save_path += '_' + str(args.max_length) + '_' + str(args.seed)
