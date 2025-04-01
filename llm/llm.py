#!/usr/bin/env python3
"""
llm.py

This script fine-tunes a small GPT-2 model (distilgpt2) on a sustainable software engineering (SSE) dataset
and performs inference to demonstrate generation capabilities.

Dependencies:
    pip install torch transformers datasets

Steps Performed:
1. Load a custom SSE text dataset.
2. Fine-tune the GPT-2 model (distilgpt2).
3. Generate a completion for a prompt using the fine-tuned model.
"""

import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset


def main():
    # === CONFIGURATION ===
    model_name = "distilgpt2"  # Pretrained small GPT-2
    output_dir = "small-llm-output"
    num_train_epochs = 2
    batch_size = 2

    # === Dataset ===
    # A list of example sentences taken from SSE course content
    text_lines = [
        "Sustainable Software Engineering integrates environmental, economic, technical, individual, and social sustainability goals.",
        "Green computing practices aim to reduce the carbon footprint of digital services and infrastructures.",
        "Energy efficiency in software is crucial to achieve a sustainable digital future.",
        "Developers are encouraged to integrate energy metrics into their testing and development cycles.",
        "Accurate energy consumption measurement can be performed using hardware power monitors and software energy profilers.",
        "Standardized energy metrics help quantify the environmental impact of software applications.",
        "Green Software Metrics include measures such as energy in joules, power in watts, and carbon emissions in grams.",
        "Carbon-aware datacenters optimize workload scheduling based on real-time carbon intensity data.",
        "The concept of red AI emphasizes maximizing model accuracy at the expense of increased energy and financial costs.",
        "Green AI seeks to balance high model performance with lower energy consumption and environmental impact."#,
        # "Techniques such as model simplification, hyperparameter tuning, and distillation can help reduce energy usage in AI training.",
        # "A Firefox extension can provide real-time feedback on the estimated carbon emissions of AI models.",
        # "HuggingFace’s AutoTrain feature automatically reports CO2 equivalent emissions during model training.",
        # "Linear regression models can be used to estimate carbon emissions based on parameters like dataset size and model size.",
        # "A shift toward efficiency over raw accuracy is vital for sustainable AI research.",
        # "Reporting carbon emissions transparently is essential for accountability in AI development.",
        # "The carbon intensity of electricity varies by energy source, with renewables generally having lower values than fossil fuels.",
        # "Code Carbon is a tool that estimates CO2 emissions based on measured energy consumption.",
        # "Benchmarking energy consumption requires careful experimental design and statistical analysis.",
        # "Violin plots and box plots are effective visual tools to understand the distribution of energy usage data.",
        # "Multiple iterations of energy experiments improve the reliability of consumption measurements.",
        # "Reproducibility and control of confounding factors are key in scientific energy measurement experiments.",
        # "Scientific guides provide methodologies for reliable energy experiments in software systems.",
        # "Statistical significance tests, like the Shapiro-Wilk and Mann-Whitney U tests, help validate energy data.",
        # "Energy Delay Product (EDP) is a combined metric that penalizes long runtimes and high energy consumption.",
        # "The integration of energy testing into CI/CD pipelines can help track efficiency improvements over time.",
        # "Efficient data structures and algorithms can lower CPU usage and overall energy consumption.",
        # "Software refactoring aimed at reducing energy consumption can also extend the battery life of mobile devices.",
        # "Green computing practices are becoming an important competitive factor for environmentally conscious companies.",
        # "The social dimension of sustainability includes ensuring equitable access to green technologies.",
        # "Technical sustainability focuses on building software systems that are resilient and adaptable over time." #,
        # "Economic sustainability in software engineering involves cost-efficient development and operation.",
        # "Environmental sustainability in software refers to minimizing the ecological footprint through efficient design.",
        # "Green procurement encourages choosing energy-efficient hardware and software solutions.",
        # "Developers should consider energy efficiency when selecting libraries and designing system architectures.",
        # "Cloud computing providers are exploring dynamic resource allocation to reduce energy usage.",
        # "Virtualization and containerization can improve resource utilization and lower power consumption.",
        # "Approximate computing allows trading off a little accuracy for significant energy savings.",
        # "ApproxSciMate is a Python library that approximates SciPy functions to reduce computational cost and energy usage.",
        # "Energy efficiency metrics enable developers to quantify the benefits of code optimizations.",
        # "Incorporating energy measurement into regression tests helps detect energy regressions early.",
        # "Tools like Intel Performance Counter Monitor provide insights into CPU energy consumption.",
        # "Monsoon power monitors and PyMonsoon libraries enable precise energy measurements on small devices.",
        # "Jupyter Notebooks, widely used in data science, can be extended to include energy measurement capabilities.",
        # "Integrating energy profiling into Jupyter workflows promotes a culture of sustainable computing.",
        # "Energy Consumption Reporter tools allow per-function energy usage tracking for detailed analysis.",
        # "Real-time energy monitoring can provide immediate feedback for energy-aware programming.",
        # "Energy consumption data can be converted into familiar units like watt-hours for easier interpretation.",
        # "Visualization of energy data using time-series and violin plots helps in understanding usage patterns.",
        # "Automated energy profiling tools can be integrated with testing frameworks like pytest.",
        # "Energy tests should account for background noise to isolate the energy usage of specific code segments.",
        # "Robust energy experiments require repeated measurements to average out transient fluctuations.",
        # "Green software engineering emphasizes reducing unnecessary computations to lower energy use.",
        # "Optimizing code for energy efficiency not only benefits the environment but can also reduce operational costs.",
        # "The exponential growth of machine learning has led to increased energy demands and carbon emissions.",
        # "Large language models, such as GPT-2 and GPT-3, demonstrate the trade-off between model size and environmental cost.",
        # "Efficient AI models are those that achieve comparable performance with fewer parameters and less energy.",
        # "The concept of green data-centric AI focuses on improving data quality to achieve energy savings.",
        # "Energy consumption is a key metric when comparing the sustainability of different AI architectures.",
        # "Green architectural tactics for ML include using energy-efficient hardware and scheduling workloads during low-carbon hours.",
        # "Statistical modeling of energy consumption can reveal insights into the relationship between compute usage and carbon output.",
        # "Sustainable software development educates engineers on the environmental implications of their design choices.",
        # "Universities are incorporating sustainable software engineering topics into their curricula.",
        # "Research in green software often involves both empirical studies and theoretical analyses of energy consumption.",
        # "The global ICT sector is responsible for a significant and growing percentage of worldwide energy usage.",
        # "Reducing energy consumption in software systems contributes to lowering global carbon emissions.",
        # "Energy-aware software development practices can drive down energy consumption across data centers.",
        # "Standardized APIs for energy measurement enable consistent reporting and comparison across projects.",
        # "Advanced energy profilers are needed to capture granular power usage data for different software components.",
        # "Software energy metrics can guide decision-making during the design, implementation, and optimization phases.",
        # "An energy-efficient coding paradigm involves balancing performance with reduced power consumption.",
        # "Developers can use energy data to identify bottlenecks and optimize resource-intensive code segments.",
        # "The integration of green metrics into software performance reports promotes sustainable practices.",
        # "SustainableSE courses combine theory and practice to teach eco-friendly software engineering techniques.",
        # "Courses in Sustainable Software Engineering cover topics ranging from energy measurement to green coding practices.",
        # "Practical labs on measuring energy consumption provide hands-on experience with energy profilers and monitors."
    ]

    dataset = Dataset.from_dict({"text": text_lines})

    # === Tokenization ===
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    # === Data Collator ===
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # === Model Loading ===
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # === Training Configuration ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=5,
        logging_dir="logs",
        do_train=True,
        do_eval=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset
    )

    # === Training ===
    print("Starting finetuning on SSE dataset...")
    trainer.train()
    print("Finetuning complete!")

    # === Inference ===
    prompt_text = "Green AI seeks to balance high model performance with"
    prompt_enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

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
