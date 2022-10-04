# Load the model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
 
load_model = AutoModelForSequenceClassification.from_pretrained("models/FineTunedEmoEvent")
 
load_tokenizer = AutoTokenizer.from_pretrained("models/FineTunedEmoEvent")

from transformers import pipeline
my_pipeline  = pipeline("sentiment-analysis", model=load_model, tokenizer=load_tokenizer)
data = ["estoy muy disgustado"]
 
print(my_pipeline(data))