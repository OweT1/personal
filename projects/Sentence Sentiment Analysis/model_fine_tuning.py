import pandas as pd
from model_training import train_model, train_lora_model

training_data = pd.read_csv('data/training_data.csv')
validation_data = pd.read_csv('data/validation_data.csv')
training_model_dir = "google-bert/bert-large-uncased"
# training_model_dir = "google-bert/bert-base-uncased"
# training_model_dir = "distilbert/distilbert-base-uncased"
num_epochs = 20
num_labels = 2 # Only 2 labels - Positive or Negative
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Trains model without PEFT
# train_model(training_data=training_data, 
#             validation_data=validation_data, 
#             model_dir=training_model_dir,
#             num_epochs=num_epochs,
#             num_labels=num_labels,
#             id2label=id2label,
#             label2id=label2id
#             )

# Trains model with PEFT (LoRA and Quantisation)
train_lora_model(training_data=training_data, 
            validation_data=validation_data, 
            model_dir=training_model_dir,
            num_epochs=num_epochs,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
            )