#!/usr/bin/env python3
"""
inference_llm.py

This script loads the finetuned GPT-2 model saved from finetune_llm.py and performs inference (text generation).

Requires:
    pip install transformers[torch]
    pip install torch
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    # Directory where the finetuned model is saved
    model_dir = "small-llm-output"

    # Load the finetuned model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    # Define a prompt for inference
    prompt_text = "Environmental sustainability in software refers"
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = prompt_enc["input_ids"]
    attention_mask = prompt_enc["attention_mask"]

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Input prompt:", prompt_text)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()
