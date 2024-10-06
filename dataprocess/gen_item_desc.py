import sys
# sys.path.append('/data1/meisen/TASTE-main')

import json
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer

from utils import load_item_name, load_item_address
from tqdm import tqdm
import pickle as pkl


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='Amazon', help='choose {Amazon} or {yelp}')
    parser.add_argument('--data_path', type=str,
                        default='/codes/l/PTGCN-main/data/beauty'
                        )
    parser.add_argument('--item_file', type=str,
                        default='/codes/l/PTGCN-main/data/beauty/Amazon_Beauty.item',
                        help='Path of the item.txt file')
    parser.add_argument('--output', type=str, default='item_desc.pkl')
    parser.add_argument('--output_dir', type=str, default='/codes/l/PTGCN-main/data/beauty',
                        help='Output data path.')
    parser.add_argument('--tokenizer', type=str,
                        default='/codes/share/huggingface_models/bert_base_uncased')
    parser.add_argument('--item_size', type=int, default=64,
                        help='maximum length of tokens of item text')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    
    item_desc = load_item_name(args.item_file)
    output_file = os.path.join(args.output_dir, args.output)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.data_path, 'user_ids_invmap.pkl'), 'rb') as f:
        user_ids_invmap = pkl.load(f)
    with open(os.path.join(args.data_path, 'item_ids_invmap.pkl'), 'rb') as f:
        item_ids_invmap = pkl.load(f)
    # with open(output_file, 'w') as f:
    #     for id, item in tqdm(item_desc.items(), desc="Tokenize item text"):
    #         group = {}
    #         item_ids = tokenizer.encode(item, add_special_tokens=False, padding=False, truncation=True, max_length=args.item_size)
    #         try:
    #             group['id'] = item_ids_invmap[id]
    #         except:
    #             continue
    #         group['item_ids'] = item_ids
    #         f.write(json.dumps(group) + '\n')
    group = {}
    for id, item in tqdm(item_desc.items(), desc="Tokenize item text"):
        try:
            node_id = item_ids_invmap[id]
            group[node_id] = item
            # item_desc = tokenizer(item, return_tensors='pt', padding=True, truncation=True, max_length=args.item_size)
            # group[node_id] = item_desc
        except:
            continue
    # print(group)
    group_sorted = sorted(group.items(), key=lambda x: x[0])
    group = dict(group_sorted)
    group_list = list(group.values())
    # print(group_list[1])
    # print(group[1])
    # print(group_list[1] == group[1])
    for id, item in enumerate(group_list):
        try:
            assert item == group[id]
        except:
            print(id, item)
            print(group[id])
            exit()
    item_descs = tokenizer(group_list, return_tensors='pt', padding=True, truncation=True, max_length=args.item_size)
    # with open(output_file, 'wb') as f:
    #     pkl.dump(group, f)
    with open(output_file, 'wb') as f:
        pkl.dump(item_descs, f)
    print('-----finish------')


if __name__ == '__main__':
    main()