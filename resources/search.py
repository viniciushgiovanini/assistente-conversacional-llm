from transformers import BertTokenizer, BertModel, BartTokenizer, BartModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pypdf import PdfReader 
import re
import time

# Exemplo de array de textos
# texts = [
#     "Este é um exemplo de texto.",
#     "Python é uma linguagem de programação popular.",
#     "Transformers são modelos poderosos de machine learning."
# ]


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
      # if i == 4:
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
# Pergunta que queremos pesquisar
question = "Esse texto atinge o públio letrado em geral ?"
pdf_path = "../data/12.História do Brasil - Boris Fausto (Colônia).pdf"

# texts = clean_string_array(read_pdf(pdf_path=pdf_path))


# Salvar o array em um arquivo
# array = np.array(texts)
# np.save('../data/livro_tokenizado.npy', array)
texts = np.load('../data/livro_tokenizado.npy')


# Carregar tokenizer e modelo BERT pré-treinado
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Tokenizar os textos e a pergunta
tokenized_texts = [tokenizer(text, return_tensors='pt') for text in texts]
tokenized_question = tokenizer(question, return_tensors='pt')


# Obter os embeddings dos textos e da pergunta
with torch.no_grad():
    text_embeddings = [model(**text)['pooler_output'] for text in tokenized_texts]
    question_embedding = model(**tokenized_question)['pooler_output']

# Calcular a similaridade de cosseno entre a pergunta e cada texto
similarities = [cosine_similarity(question_embedding.numpy().reshape(1, -1), text_embedding.numpy().reshape(1, -1))[0][0] for text_embedding in text_embeddings]

# Encontrar o índice do texto com maior similaridade
most_similar_index = np.argmax(similarities)
most_similar_text = texts[most_similar_index]

print(f"Pergunta: {question}")
print(f"Texto mais similar: {most_similar_text}")
print(f"Similaridade: {similarities[most_similar_index]}")
fim = time.time()
tempo_execucao = fim - inicio
print(f"Rodou em levou {tempo_execucao / 60} minutos para executar.")