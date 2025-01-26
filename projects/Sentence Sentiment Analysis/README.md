# Sentence Sentimental Analysis 😀😞

## Project Overview 📊
The project is simply about analysing the sentimental meaning of sentences, which are defined to be either positive or negative. In our case, we will make use of Large Language Models (LLM) and fine-tune them based on the data provided, utilising Parameter-Efficient Fine-Tuning (PEFT) methods, in this case Low Rank Adaption (LoRA) and Quantisation to train our models in a faster way.

## Project Tech Stack 👨‍💻
Programming Language: Python
Libraries & Dependencies: `transformers`, `peft`, `bitsandbytes`, `torch`, `torchvision`, `torchaudio`, `accelerate`, `seaborn`, `matplotlib`, `scikit-learn`, `numpy`, `pandas` (Can be found in the `requirements.txt` file)
Others: Hugging Face

## Brief Summary of Models Tested 📝
The details and results of the various models tested are below:

| Base Model                       | Epochs | Precision | Accuracy | Remarks                             | Model Link*                                                                                                                                                     |
|----------------------------------|--------|-----------|----------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `google-bert/bert-base-uncased`  | 20     |           |          |                                     | [https://huggingface.co/OwenTanKL/bert-base-uncased-finetuned-sa-20epochs](https://huggingface.co/OwenTanKL/bert-base-uncased-finetuned-sa-20epochs)           |
| `google-bert/bert-large-uncased` | 20     | 0.91      | 0.88     | Trained using LoRA and Quantisation | [https://huggingface.co/OwenTanKL/bert-large-uncased-finetunedlora-sa-20epochs](https://huggingface.co/OwenTanKL/bert-large-uncased-finetunedlora-sa-20epochs) |
|                                  |        |           |          |                                     |                                                                                                                                                                |

*As Github is unable to accomodate our model files due to the file size capacity, I have uploaded them onto my Hugging Face account instead. Hence, the code may not work as intended if you were to try it out yourself.

