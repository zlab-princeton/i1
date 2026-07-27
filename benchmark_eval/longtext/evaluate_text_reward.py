import os
import glob
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

from tqdm import tqdm
from datetime import timedelta

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def clean_and_remove_hallucinations(text):
    # keywords_list can be added to process ocr results to a cleaner version
    keywords_list = ["addCriterion", "No text recognized."] 
    for keyword in keywords_list:
        s = text.replace(keyword, "").replace(f"\n{keyword}", "").replace(f"{keyword}\n", "")

    return s


class ImageEvaluator:
    def __init__(self, device):
        self.device = device

        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.qwen_processor = AutoProcessor.from_pretrained(model_id)
        self.qwen_prompt = (
            "Recognize the text in the image, only reply with the text content, "
            "but avoid repeating previously mentioned content. "
            "If no text is recognized, please reply with 'No text recognized'."
        )

    def qwen_ocr(self, image):
        message = [{"role": "user",
                    "content": [{"type": "image", "image": image},
                                {"type": "text", "text": self.qwen_prompt}],}]
        texts = self.qwen_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(message)
        inputs = self.qwen_processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = self.qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        qwen_results = clean_and_remove_hallucinations(outputs[0])
        return qwen_results

    def evaluate(self, data_chunk):
        assert len(data_chunk) >= 1, f'At least one document is required.'
        eval_results = []
        with torch.no_grad():
            for data in tqdm(data_chunk):
                ocr_results = self.qwen_ocr(data['image'])
                final_result = {'image': data['image'],
                                'prompt': data['prompt'],
                                'ocr_gt': data['text'],
                                'ocr_results': ocr_results}
                eval_results.append(final_result)

        return eval_results


def split_list(x, n):
    k, m = divmod(len(x), n)
    return [x[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main(args):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl", timeout=timedelta(hours=1))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)

    os.makedirs(args.output_dir, exist_ok=True)

    prompt_file = 'text_prompts.jsonl' if args.mode == 'en' else 'text_prompts_zh.jsonl'
    prompts = [json.loads(line) for line in open(prompt_file)]
    prompt_map = {p['prompt_id']: p for p in prompts}

    data = []
    for image_file in sorted(glob.glob(f'{args.sample_dir}/*.png')):
        fname = image_file.split('/')[-1]
        if '_' in fname:
            prompt_id = int(fname.split('_')[0])
        else:
            prompt_id = int(fname.split('.')[0])
        info = prompt_map[prompt_id]
        data.append({'image': image_file, 
                     'prompt': info['prompt'], 
                     'text': info['text']})
    data_chunks = split_list(data, world_size)

    evaluator = ImageEvaluator(device)
    print(f'=============Evaluate {len(data_chunks[rank])} images in rank {rank}=============')
    results = evaluator.evaluate(data_chunks[rank])
    output_path = os.path.join(args.output_dir, f'results_chunk{rank}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['en', 'zh'], default='en')
    parser.add_argument("--global_seed", type=str, default=42)
    
    args = parser.parse_args()
    main(args)
