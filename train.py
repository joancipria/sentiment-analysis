#!pip install transformers
#!pip install datasets
#!pip install numpy
#!pip install pandas
#!pip install torch
#!pip install sklearn

from datasets import load_dataset

dataset = load_dataset("EmoEvent", use_auth_token=True)

dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")

def tokenize_function(examples):
    return tokenizer(examples["tweet"], padding="max_length", truncation=True)
 
tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import AutoModelForSequenceClassification
checkpoint = "PlanTL-GOB-ES/roberta-base-bne"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=7)

import numpy as np
from datasets import load_metric
 
metric = load_metric("accuracy")
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer
 
training_args = TrainingArguments(output_dir="trainer", evaluation_strategy="epoch", num_train_epochs=10)
 
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)
 
trainer.train()

model.save_pretrained("models/FineTunedEmoEvent")
 
# alternatively save the trainer
# trainer.save_model("models/FineTunedEmoEvent")
 
tokenizer.save_pretrained("models/FineTunedEmoEvent")