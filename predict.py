from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# Predict sentiment for the following text
texts = [
    "me siento mal porque le he pegado a mi novia el vih",
    "estoy destrozado, me han detectado vih",
    "he contraido el puto vih",
    "estoy bien jodido, me han pegado el sida",
    "he pillado una puta ets",
    "me han pegado la hepatitis",
    "estoy hasta los cojones de medicarme",
    "me encuentro genial con la nueva medicación",
    "me ecuentro bastante bien con ese medicamento"
]

for text in texts:
    print("-----------")
    print("Predict: ", '"', text, '"')

    # Select model
    FINE_TUNED_MODELS = [
        "models/PlanTL-GOB-ES/gpt2-base-bne-FineTunedEmoEvent",
        "models/PlanTL-GOB-ES/gpt2-large-bne-FineTunedEmoEvent",
        "models/PlanTL-GOB-ES/roberta-base-biomedical-es-FineTunedEmoEvent",
        "models/PlanTL-GOB-ES/roberta-base-bne-FineTunedEmoEvent",
        "models/PlanTL-GOB-ES/roberta-large-bne-FineTunedEmoEvent",
    ]

    for FINE_TUNED_MODEL in FINE_TUNED_MODELS:
        # Load the model
        load_model = AutoModelForSequenceClassification.from_pretrained(
            FINE_TUNED_MODEL)

        load_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL)

        # Setup pipeline
        sentiment_pipeline = pipeline("sentiment-analysis",
                                      model=load_model, tokenizer=load_tokenizer)

        # Get sentiment
        sentiment_dump = sentiment_pipeline(text)
        sentiment_label = sentiment_dump[0]["label"]

        if sentiment_label == 'LABEL_6':
            sentiment = "other"
        elif sentiment_label == 'LABEL_5':
            sentiment = "surprise"
        elif sentiment_label == 'LABEL_4':
            sentiment = "disgust"
        elif sentiment_label == 'LABEL_3':
            sentiment = "joy"
        elif sentiment_label == 'LABEL_2':
            sentiment = "sadness"
        elif sentiment_label == 'LABEL_1':
            sentiment = "fear"
        elif sentiment_label == 'LABEL_0':
            sentiment = "anger"
        else:
            sentiment = "unknown"

        print("With: ", FINE_TUNED_MODEL, "➔",
              sentiment, sentiment_dump[0]["score"])
