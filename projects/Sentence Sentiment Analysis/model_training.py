import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig, TrainingArguments, AutoModelForSequenceClassification, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from model_extraction import get_tokenizer, get_data_collator

def tokenize_input(text, label, tokenizer):
    tokenized_text = tokenizer(text)
    tokenized_text['label'] = label
    return tokenized_text

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='macro'),
        'precision': precision_score(labels, predictions, average='macro'),
        'recall': recall_score(labels, predictions, average='macro')
    }

def train_model(training_data, validation_data, model_dir, num_epochs, num_labels, id2label, label2id):
    tokenizer = get_tokenizer(model_dir)
    data_collator = get_data_collator(tokenizer)
    train_dataset = training_data.apply(lambda x: tokenize_input(text=x['Text'], label=x['Label'], tokenizer=tokenizer), axis=1)
    eval_dataset = validation_data.apply(lambda x: tokenize_input(text=x['Text'], label=x['Label'], tokenizer=tokenizer), axis=1)

    model_name = model_dir.split('/')[1]
    output_dir = f"models/{model_name}_finetuned_{num_epochs}epochs"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        save_total_limit=1,
        num_train_epochs=num_epochs,
        warmup_steps=50)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    print(trainer.evaluate(eval_dataset=eval_dataset))
    trainer.save_model(output_dir=f"{output_dir}_best")
    print(f"Best model saved at: {output_dir}_best")

def train_lora_model(training_data, validation_data, model_dir, num_epochs, num_labels, id2label, label2id):
    tokenizer = get_tokenizer(model_dir)
    data_collator = get_data_collator(tokenizer)
    train_dataset = training_data.apply(lambda x: tokenize_input(text=x['Text'], label=x['Label'], tokenizer=tokenizer), axis=1)
    eval_dataset = validation_data.apply(lambda x: tokenize_input(text=x['Text'], label=x['Label'], tokenizer=tokenizer), axis=1)

    model_name = model_dir.split('/')[1]
    output_dir = f"models/{model_name}_finetunedlora_{num_epochs}epochs"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, # enable 4-bit quantization
        bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
        bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
        bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, 
        num_labels=num_labels,
        quantization_config=quantization_config,
        id2label=id2label,
        label2id=label2id
        )

    # lora config
    lora_config = LoraConfig(
        r = 32, # the dimension of the low-rank matrices
        lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
        target_modules = 'all-linear', 
        lora_dropout = 0.1, # dropout probability of the LoRA layers
        bias = 'none', # whether to train bias weights, set to 'none' for attention layers
        task_type = 'SEQ_CLS',
        use_rslora = True
    )

    model = prepare_model_for_kbit_training(model)
    lora_model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        save_total_limit=1,
        num_train_epochs=num_epochs,
        warmup_steps=50)

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    print(trainer.evaluate(eval_dataset=eval_dataset))
    trainer.save_model(output_dir=f"{output_dir}_best")
    print(f"Best model saved at: {output_dir}_best")