import argparse
import json
import os
import sys
sys.path.append('extlib')

import torch
from PIL import Image
from tqdm import tqdm

from extlib.hycoclip import lorentz as L

from model_zoo import get_model



def load_model(args, pretrained, device):
    model, preprocess = get_model(args.model, pretrained, device)
    tokenizer = model.tokenizer
    model = model.model.eval()
    return model, tokenizer, preprocess


@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model, tokenizer, preprocess, device):
    pos_text = [t.to(device) for t in tokenizer(pos_text)]
    pos_text_embedding = model.encode_text(pos_text, project=True)
    neg_text = [t.to(device) for t in tokenizer(neg_text)]
    neg_text_embedding = model.encode_text(neg_text, project=True)
    image_embedding = model.encode_image(preprocess(image).unsqueeze(dim=0).to(device), project=True)
    
    if hasattr(model, 'curv'):
        curv = model.curv.exp()
        pos_score = L.pairwise_inner(pos_text_embedding, image_embedding, curv)
        neg_score = L.pairwise_inner(neg_text_embedding, image_embedding, curv)
    else:
        pos_score = pos_text_embedding @ image_embedding.t()
        neg_score = neg_text_embedding @ image_embedding.t()
        
    return 1 if pos_score.item() > neg_score.item() else 0


def evaluate(image_root, dataset, model, tokenizer, preprocess, device):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            image_path = os.path.join(image_root, data['filename'])
            image = Image.open(image_path)
            correct = text_retrieval(data['caption'], data['negative_caption'], image, model, tokenizer, preprocess, device)
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="RN50", help="Model architecture to use from OpenCLIP")
    parser.add_argument('--pretrained', type=str, default="openai", help="Model checkpoint name to use from OpenCLIP")
    parser.add_argument('--output', type=str, default=None, help="Directory to where results are saved")

    parser.add_argument('--coco_image_root', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dict = {
        'add_obj'    : f'{args.data_root}/add_obj.json',
        'add_att'    : f'{args.data_root}/add_att.json',
        'replace_obj': f'{args.data_root}/replace_obj.json',
        'replace_att': f'{args.data_root}/replace_att.json',
        'replace_rel': f'{args.data_root}/replace_rel.json',
        'swap_obj'   : f'{args.data_root}/swap_obj.json',
        'swap_att'   : f'{args.data_root}/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))

    os.makedirs(args.output, exist_ok=True)

    print(f"Evaluating {args.model}-{args.pretrained}")

    model, tokenizer, transform = load_model(args, args.pretrained, device)

    metrics = evaluate(args.coco_image_root, dataset, model, tokenizer, transform, device)
    print(metrics)
