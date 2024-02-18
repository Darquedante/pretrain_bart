# Full source code for data_preprocessing.py with enhancements and dataset integration

import argparse
import logging
import random
import re
import sys
from itertools import groupby

import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def parse_args(args):
    parser = argparse.ArgumentParser(description="Preprocess data for BART language model training with masking and shuffling.")
    parser.add_argument("--data", type=str, help="Path to the training data file or dataset name from Hugging Face datasets (if --use_hf_dataset is set).")
    parser.add_argument("--tokenizer_config", type=str, default="facebook/mbart-large-50-one-to-many-mmt", help="Tokenizer configuration for preprocessing.")
    parser.add_argument("--output_name", type=str, default="bart_pretrain_data", help="Base name for output files.")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of masking a token.")
    parser.add_argument("--worker", type=int, default=10, help="Number of workers for parallel processing.")
    parser.add_argument("--poisson_lam", type=int, default=3, help="Lambda for Poisson distribution in masking.")
    parser.add_argument("--use_hf_dataset", action="store_true", help="Flag to use dataset from Hugging Face datasets instead of a local file.")
    return parser.parse_args(args)

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?。]) +', text)
    return sentences

def shuffle_and_mask(sentences, mask_token, mask_prob, poisson_lam):
    random.shuffle(sentences)
    masked_sentences = []
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            if random.random() < mask_prob:
                length = np.random.poisson(lam=poisson_lam)
                mask_sequence = [mask_token] * max(1, length)
                words[i:i+1] = mask_sequence
        masked_sentences.append(" ".join(words))
    return " ".join(masked_sentences)

def preprocess_function(examples, tokenizer, mask_prob, poisson_lam):
    mask_token = tokenizer.mask_token
    processed_examples = {'input_text': [], 'target_text': []}
    for doc in examples["text"]:
        sentences = split_sentences(doc)
        input_text = shuffle_and_mask(sentences, mask_token, mask_prob, poisson_lam)
        processed_examples['input_text'].append(input_text)
        processed_examples['target_text'].append(doc)
    return processed_examples

def main():
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_config)

    if args.use_hf_dataset:
        logging.info(f"Loading dataset from Hugging Face datasets: {args.data}")
        dataset = load_dataset(args.data)
    else:
        logging.info(f"Loading dataset from local file: {args.data}")
        dataset = load_dataset("text", data_files={'train': args.data})

    processed_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.mask_prob, args.poisson_lam),
        batched=True,
        num_proc=args.worker
    )

    output_path = args.output_name + "_cache"
    logging.info(f"Saving processed dataset to {output_path}...")
    processed_dataset.save_to_disk(output_path)

if __name__ == "__main__":
    main()
#python data_preprocessing.py --data your_dataset_path_or_name --use_hf_dataset (if using Hugging Face datasets)
