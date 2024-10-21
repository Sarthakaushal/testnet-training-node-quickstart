import os, sys
from dataclasses import dataclass

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
from merge import merge_lora_to_base_model
from utils.constants import model2template


@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int

def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments, revision: str
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    if "Phi-3.5-mini-instruct" in model_id or "Phi-3-mini-4k-instruct" in model_id:
        target_modules = ['gate_up_proj', 'down_proj', 'qkv_proj', 'o_proj']
        lora_config = LoraConfig(
            r=training_args.lora_rank,
            target_modules=target_modules,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            task_type="CAUSAL_LM",

        )

    else:
        target_modules = ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=training_args.lora_rank,
            target_modules=target_modules,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            task_type="CAUSAL_LM",
        )
    # Todo look for 8 bit quantization
    # Loda model in 8-bit to qLoRA
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,  # Change from 4bit to 8bit
    #     bnb_8bit_quant_type="nf8",  # If needed, but "nf8" might not exist. Use "normal" quant if available.
    #     bnb_8bit_compute_dtype=torch.bfloat16,  # You can use `torch.float16` or `torch.bfloat16`
    # )
    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # ToDo :  Look into better training configs
    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length
    )
    
    qLora_Training_args = SFTConfig(
         per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_ratio=0.03, # as per qLoRA paper
        learning_rate=2e-5,# as per qLoRA paper
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
        # New args added
        max_grad_norm=0.3 # as per qLoRA paper
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
        revision=revision
    )
    
    # Load dataset
    dataset = SFTDataset(
        file="demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )
    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=qLora_Training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # upload lora weights and tokenizer
    print("Training Completed.")
