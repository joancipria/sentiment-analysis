# Load EmoEvent dataset
from datasets import load_dataset

dataset = load_dataset("datasets/EmoEvent", use_auth_token=True)

# List of spanish pre-trained models to fine-tune
pre_trained_models = ["PlanTL-GOB-ES/roberta-base-bne", "PlanTL-GOB-ES/roberta-large-bne",
                      "PlanTL-GOB-ES/roberta-base-biomedical-es", "PlanTL-GOB-ES/gpt2-base-bne", "PlanTL-GOB-ES/gpt2-large-bne"]

# For each pre-trained model, fine-tune it
for PRE_TRAINED_MODEL in pre_trained_models:

    # Tokenize dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(examples):
        return tokenizer(examples["tweet"], padding="max_length", truncation=True, max_length=40)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load pre-trained model
    from transformers import AutoModelForSequenceClassification
    checkpoint = PRE_TRAINED_MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=7)
    model.resize_token_embeddings(len(tokenizer))

    # Define evaluation metrics
    import numpy as np
    from datasets import load_metric

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Create trainer
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="trainer", 
        evaluation_strategy="epoch", 
        num_train_epochs=5,
        learning_rate=2e-5, # Default: 5e-5.  The initial learning rate for AdamW optimizer.
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save model
    fine_tuned_model_output = "models/"+PRE_TRAINED_MODEL+"-FineTunedEmoEvent"
    model.save_pretrained(fine_tuned_model_output)
    # alternatively save the trainer
    # trainer.save_model("models/FineTunedEmoEvent")
    tokenizer.save_pretrained(fine_tuned_model_output)
