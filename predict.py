from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# Select model
FINE_TUNED_MODEL = "models/PlanTL-GOB-ES/roberta-large-bne-FineTunedEmoEvent"

# Load the model
load_model = AutoModelForSequenceClassification.from_pretrained(
    FINE_TUNED_MODEL)

load_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL)

# Setup pipeline
my_pipeline = pipeline("sentiment-analysis",
                       model=load_model, tokenizer=load_tokenizer)

# Predict sentiment for the following text
text = ["estoy muy disgustado"]
print(my_pipeline(text))
