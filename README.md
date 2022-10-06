# Sentiment Analysis
Fine-tuned [spanish language models](https://github.com/PlanTL-GOB-ES/lm-spanish) with [EmoEvent dataset](https://github.com/fmplaza/EmoEvent). A project for [VIHrtual-App](https://github.com/joancipria/VihrtualApp) chatbot.



## 📦 Install
Tested with `Python 3.10`

Clone repository
```
git clone https://github.com/joancipria/sentiment-analysis
cd sentiment-analysis
```

Create virtual environment
```
python -m venv ./venv
source ./venv/bin/activate
```

Install requirements
```
pip install -r requirements.txt
```

## 🤖 Run

Run `python train.py` to fine-tune the model.

Run `python predict.py` to test recognition.
