import os
import sys
from typing import List, Union

import fire
import torch

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
)

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 1024,
        val_set_size: int = 0,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        use_rslora: bool = True,
#        init_lora_weights: str = None,
):

    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
    )

    gradient_accumulation_steps = batch_size // micro_batch_size
    # device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:
        # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # print(f'os.environ.get("LOCAL_RANK"): {os.environ.get("LOCAL_RANK")}')
    # print(f"device_map: {device_map}")

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Maximum number of checkpoint
    save_total_limit = 3
    
    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Learning rate schedule
    lr_scheduler_type = "cosine"

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_8bit,
        bnb_4bit_compute_dtype='float16')

    # Update config while loading base model and configuration (config.json) from huggingface.co and cache
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        use_cache=False,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_target_modules,
        use_rslora=use_rslora
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    training_arguments = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_step if val_set_size > 0 else None,
        save_steps=save_step,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        max_grad_norm=max_grad_norm,
        load_best_model_at_end=True if val_set_size > 0 else False,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
    )

#==============================================================
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_prompt(data_point):
        return f"""{data_point["instruction"]}
                    ### Answer:{data_point["output"]}""" 

    data = load_dataset("json", data_files=data_path)
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    
    trainer = Trainer(
        model = lora_model,
        train_dataset=train_data,
        args=training_arguments,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
#==============================================================

    trainer.train()

    lora_model.save_pretrained(output_dir)

    return


if __name__ == "__main__":
    fire.Fire(train)
