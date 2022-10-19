from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset

# Select model
FINE_TUNED_MODEL = "models/PlanTL-GOB-ES/roberta-large-bne-FineTunedEmoEvent"


#
dataset = load_dataset("EmoEvent", use_auth_token=True)


tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def tokenize_function(examples):
    return tokenizer(examples["tweet"], padding="max_length", truncation=True, max_length=40)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the model

load_model = AutoModelForSequenceClassification.from_pretrained(
    FINE_TUNED_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL)

load_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL)

# --------------


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    load_precision = load_metric('precision')
    load_recall = load_metric('recall')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = load_accuracy.compute(
        predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions,
                         references=labels, average='weighted')["f1"]
    precision = load_precision.compute(
        predictions=predictions, references=labels, average='weighted')["precision"]
    recall = load_recall.compute(
        predictions=predictions, references=labels, average='weighted')["recall"]

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# --------------


training_args = TrainingArguments(
    output_dir="trainer", evaluation_strategy="epoch", num_train_epochs=10)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)


print(trainer.evaluate())
