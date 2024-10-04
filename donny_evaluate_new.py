import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorWithPadding
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import classification_report
import evaluate
import numpy as np

def main(
        base_model: str = "",
        weights_dir: str = "",
        output_dir: str = "",
        eval_dataset: str = "",
        load_8bit : bool = False,
):
    print(f"base_model: {base_model}")
    print(f"weights_dir: {weights_dir}")
    print(f"output_dir: {output_dir}")
    print(f"eval_dataset: {eval_dataset}")
    print(f"load_8bit: {load_8bit}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # Load model
    model = AutoPeftModelForCausalLM.from_pretrained(
        weights_dir,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    # Load Dataset
    dataset = load_dataset('json', data_files=eval_dataset, split='all')

    def generate_prompt(data_point):
        return f"""{data_point['instruction']}
                ### Answer: """ 

    def preprocess_function(example):
        #return tokenizer(generate_prompt(example), truncation=True, padding='max_length', max_length=1024)
        return tokenizer(generate_prompt(example), pad_to_multiple_of=8, return_tensors="pt", padding=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Data colator for padding
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=4,
        do_train=False,
        do_eval=True,
        logging_dir=f"{output_dir}/logs",
        evaluation_strategy="epoch",
    )

    def compute_metrics(eval_pred):
        metrics = evaluate.load('precision', 'recall', 'f1')
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        print(classification_report(labels, predictions))
        return metrics.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Run evaluation
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    return

if __name__ == "__main__":
    fire.Fire(main)