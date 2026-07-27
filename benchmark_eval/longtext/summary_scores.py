import argparse
import json
import os
import re

from tqdm import tqdm
from collections import Counter


def preprocess_string(s, mode='en'):
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", '', s)
    if mode == 'en':
        normalized = re.sub(r'\s+', ' ', cleaned)  
        return normalized.strip().lower()
    else:
        pattern = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]")
        s = ''.join(pattern.findall(s))
        return s.strip()


def counter2list(counter):
    return [item for item, count in counter.items() for _ in range(count)]


def calculate_char_match_ratio(text_gt, ocr_str, mode='en'):
    if mode == 'en':
        words_gt = text_gt.split()
        words_ocr = ocr_str.split()
        gt_counter = Counter(words_gt)
        ocr_counter = Counter(words_ocr)
        match = counter2list(gt_counter & ocr_counter)
        unmatch = counter2list(gt_counter - ocr_counter)
    else:
        words_gt = text_gt
        gt_counter = Counter(text_gt)
        ocr_counter = Counter(ocr_str)
        match = counter2list(gt_counter & ocr_counter)
        unmatch = counter2list(gt_counter - ocr_counter)
    return match, words_gt, unmatch


parser = argparse.ArgumentParser()
parser.add_argument('result_file', type=str)
parser.add_argument('--mode', type=str, choices=['en', 'zh'], default='en')
args = parser.parse_args()

result_dir = os.path.dirname(args.result_file)
results = [json.loads(l) for l in open(args.result_file)]

for r in tqdm(results):
    ocr_results = preprocess_string(r['ocr_results'], args.mode)
    ocr_gt = preprocess_string(' '.join(r['ocr_gt']), args.mode)
    
    match, gt, unmatch = calculate_char_match_ratio(ocr_gt, ocr_results, args.mode)

    r['ocr_results'] = ocr_results
    r['ocr_gt'] = ocr_gt
    r['match_word_count'] = len(match)
    r['gt_word_count'] = len(gt)
    r['text_accuray'] = len(match) / len(gt)

res_save_path = os.path.join(os.path.dirname(args.result_file), 'results.jsonl')
with open(res_save_path, 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

match_score = sum([r['match_word_count'] for r in results]) / sum([r['gt_word_count'] for r in results])
score_str = f'Text Score: {match_score:.4f}\n'
print(score_str)
output_path = os.path.join(result_dir, 'scores.txt')
with open(output_path, 'w') as f:
    f.write(score_str)
