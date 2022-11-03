# 📊 Evaluation and metrics

Evaluation performed with [STI dataset](./../datasets/sti/splits/es/test.tsv)

roberta-base-bne
```
{
    'eval_loss': 2.2949678897857666, 
    'eval_accuracy': 0.7071428571428572, 
    'eval_f1': 0.7088576439256975, 
    'eval_precision': 0.7556283002711575, 
    'eval_recall': 0.7071428571428572, 
    'eval_runtime': 6.0163, 
    'eval_samples_per_second': 23.27, 
    'eval_steps_per_second': 2.992
}
```

roberta-large-bne
```
{
    'eval_loss': 1.037258505821228, 
    'eval_accuracy': 0.7714285714285715, 
    'eval_f1': 0.7637673989315289, 
    'eval_precision': 0.8061791383219955, 
    'eval_recall': 0.7714285714285715, 
    'eval_runtime': 17.9042, 
    'eval_samples_per_second': 7.819, 
    'eval_steps_per_second': 1.005
}
```

roberta-base-biomedical-es-FineTunedEmoEvent
``` 
{
    'eval_loss': 2.03935170173645, 
    'eval_accuracy': 0.7, 
    'eval_f1': 0.6908990006846026, 
    'eval_precision': 0.7163574520717377, 
    'eval_recall': 0.7, 
    'eval_runtime': 6.6855, 
    'eval_samples_per_second': 20.941, 
    'eval_steps_per_second': 2.692
}
```

gpt2-base-bne-FineTunedEmoEvent
```
{
    'eval_loss': 2.3879334926605225, 
    'eval_accuracy': 0.6214285714285714, 
    'eval_f1': 0.6256136877311907, 
    'eval_precision': 0.7225672877846792, 
    'eval_recall': 0.6214285714285714, 
    'eval_runtime': 6.3475, 
    'eval_samples_per_second': 22.056, 
    'eval_steps_per_second': 2.836
}
```

gpt2-large-bne-FineTunedEmoEvent
```
{
    'eval_loss': 5.781673908233643, 
    'eval_accuracy': 0.4857142857142857, 
    'eval_f1': 0.45976243279914025, 
    'eval_precision': 0.5762464178173072, 
    'eval_recall': 0.4857142857142857, 
    'eval_runtime': 30.9193, 
    'eval_samples_per_second': 4.528, 
    'eval_steps_per_second': 0.582
}
```