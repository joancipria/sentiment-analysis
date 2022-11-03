# Sentiment Analysis
Fine-tuned [spanish language models](https://github.com/PlanTL-GOB-ES/lm-spanish) with [EmoEvent dataset](https://github.com/fmplaza/EmoEvent). A project for [VIHrtual-App](https://github.com/joancipria/VihrtualApp) chatbot to detect emotions in STI/HIV related conversations.

## 🤖 Models
- [RoBERTa-base-BNE-FineTunedEmovent](https://huggingface.co/joancipria/roberta-base-bne-FineTunedEmoEvent)
- [RoBERTa-large-BNE-FineTunedEmovent](https://huggingface.co/joancipria/roberta-large-bne-FineTunedEmoEvent)
- [RoBERTa-base-biomedical-es-FineTunedEmovent](https://huggingface.co/joancipria/roberta-base-biomedical-es-FineTunedEmoEvent)
- [GPT2-base-BNE-FineTunedEmovent](https://huggingface.co/joancipria/gpt2-base-bne-FineTunedEmoEvent)
- [GPT2-large-BNE-FineTunedEmovent](https://huggingface.co/joancipria/gpt2-large-bne-FineTunedEmoEvent)


## 📊 Evaluation and metrics

| Model      | F1   |  Accuracy  |
|--------------|----------|------------|
| [RoBERTa-base-BNE-FineTunedEmovent](https://huggingface.co/joancipria/roberta-base-bne-FineTunedEmoEvent)        | 0.7089       |     0.7071 |
| [RoBERTa-large-BNE-FineTunedEmovent](https://huggingface.co/joancipria/roberta-large-bne-FineTunedEmoEvent)   | 0.7638 🔥       | 0.7714 🔥    |
| [RoBERTa-base-biomedical-es-FineTunedEmovent](https://huggingface.co/joancipria/roberta-base-biomedical-es-FineTunedEmoEvent)  | 0.6909       | 0.7000     |
| [GPT2-base-BNE-FineTunedEmovent](https://huggingface.co/joancipria/gpt2-base-bne-FineTunedEmoEvent) | 0.6256       |     0.6214 |
| [GPT2-large-BNE-FineTunedEmovent](https://huggingface.co/joancipria/gpt2-large-bne-FineTunedEmoEvent)       | 0.4598       |    0.4857 |

Evaluation performed with [STI dataset](./datasets/sti/splits/es/test.tsv). See more details at [metrics docs](./docs/Metrics.md).

## ⚗️ Usage example
```python
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# Load the model
load_model = AutoModelForSequenceClassification.from_pretrained("joancipria/roberta-large-bne-FineTunedEmoEvent")

load_tokenizer = AutoTokenizer.from_pretrained("joancipria/roberta-large-bne-FineTunedEmoEvent")

# Setup pipeline
my_pipeline = pipeline("sentiment-analysis", model=load_model, tokenizer=load_tokenizer)

# Predict sentiment for the following text
text = ["me encuentro genial con la nueva medicación"]
print(my_pipeline(text))
```


## 📦 Install
Tested with `Python 3.10.7`

Clone repository
```
git clone https://github.com/joancipria/sentiment-analysis && cd sentiment-analysis
```

Create virtual environment
```
python -m venv ./venv && source ./venv/bin/activate
```

Install requirements
```
pip install -r requirements.txt
```

## ▶️ Run
Run `python train.py` to fine-tune the models.
Edit and run `python predict.py` to test recognition.
Edit and run `python eval.py` to evaluate a model.

## 📝 Cite
To cite this resource in a publication, please use the following:

```
@inproceedings{moreno2022conversational,
  title={A Conversational Agent for Medical Disclosure of Sexually Transmitted Infections},
  author={Moreno, Joan C and S{\'a}nchez-Anguix, Victor and Alberola, Juan M and Juli{\'a}n, Vicente and Botti, Vicent},
  booktitle={International Conference on Hybrid Artificial Intelligence Systems},
  pages={431--442},
  year={2022},
  organization={Springer}
}
```

## 📜 License
Licensed under GNU General Public License v3. 