from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TextClassificationPipeline
from peft import PeftModel

def get_model(model_dir, model_name, num_epochs, num_labels, id2label, label2id, lora=False):
    if lora:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        peft_model_id = f'models/{model_name}_finetunedlora_{num_epochs}epochs_best'

        model = PeftModel.from_pretrained(base_model, peft_model_id)
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            f'models/{model_name}_finetuned_{num_epochs}epochs_best',
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
    return model

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def get_data_collator(tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator

def get_pipeline(model, tokenizer):
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline