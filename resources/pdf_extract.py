from transformers import BertTokenizer, BertModel, BartTokenizer, BartModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pypdf import PdfReader 
import re
import time
import pickle

def tratar_leitura_pdf(caracteres, palavras):
  palavra_atual = []

  for char in caracteres:
      if "." not in char:
          palavra_atual.append(char)
      else:
          if palavra_atual:
              palavra = ''.join(palavra_atual)
              palavras.append(palavra)
              palavra_atual = []  

  if palavra_atual:
      palavra = ''.join(palavra_atual)
      palavras.append(palavra)
  return palavras


def read_pdf(pdf_path):
    reader = PdfReader(pdf_path) 
    page = reader.pages
    lista_por_linha = []
    
    for i, each_page in enumerate(page):
      if i >= 4 and i <= 84:
        text = ""
        each_page = reader.pages[i]
        text = each_page.extract_text()
        tratar_leitura_pdf(text, lista_por_linha)
        print(len(lista_por_linha))
    return lista_por_linha
  
def clean_string_array(arr):
  
  clean_arr = []
  
  for each in arr:
    texto_limpo = re.sub(r'[^a-zA-ZÀ-ÖØ-öø-ÿçÇ\s]', '', each)
    
    texto_limpo = texto_limpo.lower()
    
    texto_limpo = texto_limpo.replace("\n", " ")
    
    clean_arr.append(texto_limpo)
    
    
  return clean_arr



inicio = time.time()
question = "Esse texto atinge o públio letrado em geral ?"
pdf_path = "../data/12.História do Brasil - Boris Fausto (Colônia).pdf"

texts = clean_string_array(read_pdf(pdf_path=pdf_path))


with open("../data/pdf_extract_gpu.pkl", "wb") as f:
  pickle.dump(texts, f)


print(len(texts))

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')


tokenized_texts = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

model.to(device)
tokenized_texts.to(device)

with torch.no_grad():
    text_embeddings = model(**tokenized_texts)['last_hidden_state'][:, 0, :]  

with open("../models/models_gpu.pkl", 'wb') as f:
    pickle.dump(model, f)


with open("../data/text_embeddings_gpu.pkl", "wb") as f:
  pickle.dump(text_embeddings, f)

fim = time.time()
tempo_execucao = fim - inicio
print(f"Rodou em levou {tempo_execucao / 60} minutos para executar.")