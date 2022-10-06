# Load the model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
 
load_model = AutoModelForSequenceClassification.from_pretrained("models/FineTunedEmoEvent")
 
load_tokenizer = AutoTokenizer.from_pretrained("models/FineTunedEmoEvent")

# Setup pipeline
from transformers import pipeline
my_pipeline  = pipeline("sentiment-analysis", model=load_model, tokenizer=load_tokenizer)

# Predict sentiment for the following text
text = ["estoy muy disgustado"] 
print(my_pipeline(text))