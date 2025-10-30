import argparse
import sys
sys.path.append("..")

import pandas as pd
import json
import numpy as np

from torch.utils.data import DataLoader
from model_zoo import get_model
from dataset_zoo import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root of the project', default='.', type=str)
parser.add_argument('--model_name', help='Specify MERU/CLIP and S/B/L_', default='meru_vit_b', type=str)
parser.add_argument('--checkpoint', help='path to model checkpoint', default='', type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--dataset", default="VG_Relation", type=str, choices=["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order", "all"])
parser.add_argument("--seed", default=1, type=int)

parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
parser.add_argument("--output-dir", default="./outputs", type=str)
args = parser.parse_args()

root_dir = args.root + '/datasets'

model, preprocess = get_model(model_name=args.model_name, checkpoint=args.checkpoint, device="cuda", root=args.root, root_dir=root_dir)

datasets = [args.dataset] if args.dataset != 'all' else ["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"]

for dataset_name in datasets:
    dataset = get_dataset(dataset_name, image_preprocess=preprocess, download=args.download)

    joint_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Compute the scores for each test case
    scores = model.get_retrieval_scores_batched(joint_loader)
    result_records = dataset.evaluate_scores(scores)

    metric_key = 'Precision@1' if 'Order' in dataset_name else 'Accuracy'
    df = pd.DataFrame(result_records)
    print(f"{dataset_name} Macro {metric_key}: {df[metric_key].mean()}")
    df.to_csv(args.model_name +'_' + dataset_name + '_acc.csv')
