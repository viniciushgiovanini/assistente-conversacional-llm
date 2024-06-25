from transformers import BertTokenizer, BertModel, BartTokenizer, BartModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import numpy as np
import time
import pickle


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
print(f"O dispositivo utilizado atualmente é {device}")

question = "Esse texto atinge o públio letrado em geral ?"

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
  
with open("../data/text_embeddings_gpu.pkl", 'rb') as f:
    text_embeddings = pickle.load(f)

with open("../data/pdf_extract_gpu.pkl", "rb") as f:
  texts = pickle.load(f)


tokenized_texts = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

model.to(device)
tokenized_texts.to(device)

inicio = time.time()
tokenized_question = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
tokenized_question.to(device)
    
with torch.no_grad():
    question_embedding = model(**tokenized_question)['last_hidden_state'][:, 0, :] 
# question_embedding.to(device)
# similarities = [cosine_similarity(question_embedding.numpy().reshape(1, -1), text_embedding.numpy().reshape(1, -1))[0][0] for text_embedding in text_embeddings]
similarities = [F.cosine_similarity(question_embedding, text_embedding.unsqueeze(0)).item() for text_embedding in text_embeddings]

# most_similar_index = np.argmax(similarities)
# most_similar_text = texts[most_similar_index]
most_similar_index = torch.argmax(torch.tensor(similarities))
most_similar_text = texts[most_similar_index]

print(f"Pergunta: {question}")
print(f"Texto mais similar: {most_similar_text}")
print(f"Similaridade: {similarities[most_similar_index]}")
fim = time.time()
tempo_execucao = fim - inicio
print(f"Rodou em levou {tempo_execucao / 60} minutos para executar.")