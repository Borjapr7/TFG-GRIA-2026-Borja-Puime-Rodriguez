import sys
import json
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding
)

def main():
    if len(sys.argv) < 2:
        print("Error: Debes indicar la tarea GLUE.")
        sys.exit(1)
        
    task = sys.argv[1]
    
    model_name = "citiusLTL/GambBERT"
    
    configuraciones = {
        "sst2": {"keys": ("sentence", None), "num_labels": 2},
        "qqp": {"keys": ("question1", "question2"), "num_labels": 2},
        "stsb": {"keys": ("sentence1", "sentence2"), "num_labels": 1},
        "mnli": {"keys": ("premise", "hypothesis"), "num_labels": 3},
        "rte": {"keys": ("sentence1", "sentence2"), "num_labels": 2},
        "cola": {"keys": ("sentence", None), "num_labels": 2}
    }
    
    config = configuraciones[task]
    print(f"Iniciando tarea: {task.upper()} ")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["num_labels"])

    ds = load_dataset("nyu-mll/glue", task)
    def tokenize_fn(examples):
        key1, key2 = config["keys"]
        if key2 is None:
            return tokenizer(examples[key1], truncation=True, padding="max_length", max_length=128)
        else:
            return tokenizer(examples[key1], examples[key2], truncation=True, padding="max_length", max_length=128)

    tokenized_ds = ds.map(tokenize_fn, batched=True)
    metric = evaluate.load("glue", task)
    is_regression = task == "stsb"

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if is_regression:
            predictions = np.squeeze(logits)
        else:
            predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    eval_split = "validation_matched" if task == "mnli" else "validation"

    args = TrainingArguments(
        output_dir=f"./results_disorbert_{task}",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3, 
        weight_decay=0.01,
        load_best_model_at_end=False,
        fp16=True, 
        report_to="none", 
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"], 
        eval_dataset=tokenized_ds[eval_split],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    res = trainer.evaluate()
    
    if task == "cola":
        metrica = res['eval_matthews_correlation']
    elif task == "stsb":
        metrica = res['eval_spearmanr']
    elif task in ["qqp", "rte"]:
        metrica = res.get('eval_accuracy', res.get('eval_f1'))
    else:
        metrica = res['eval_accuracy']

    print(f"Resultado {task.upper()}: {metrica:.4f}")

    with open(f"resultado_{task}.json", "w") as f:
        json.dump({task: metrica}, f)

if __name__ == "__main__":
    main()