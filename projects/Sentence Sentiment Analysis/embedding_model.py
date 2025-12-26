import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

training_data = pd.read_csv('data/training_data.csv')
training_sentences = training_data['Text']
training_labels = training_data['Label']

def get_sentiments(test_sentences, training_sentences=training_sentences, training_labels=training_labels):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, # enable 4-bit quantization
        bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
        bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
        bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
    )

    # Load the NV-Embed-v2 model and tokenizer - We need to quantize the model, if not we are unable to run it due to the large amount of parameters
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', quantization_config=quantization_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2')

    # Combine all sentences
    all_sentences = training_sentences + [test_sentences]
    
    # Tokenize and generate embeddings
    inputs = tokenizer(all_sentences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Calculate cosine similarity
    test_embeddings = embeddings[len(training_sentences):]
    similarities = torch.matmul(test_embeddings, embeddings[:len(training_sentences)].T)
    
    # Calculate average similarity to positive and negative sentences
    positive_sim = similarities[training_labels].mean()
    negative_sim = similarities[~training_labels].mean()
    
    # Determine sentiments
    return positive_sim, negative_sim