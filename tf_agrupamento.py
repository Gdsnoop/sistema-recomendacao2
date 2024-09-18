# -*- coding: utf-8 -*-
"""TF_agrupamento.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qLM8HFy6XmkYNKzK-GDF-SWecR1utdD4
"""

#Importando as bibliotecas
import numpy as np
from sklearn.cluster import KMeans

#Matriz com os filmes vistos
filmes_assitidos = np.array([
    [1,0,0,1],
    [1,1,0,0],
    [0,1,1,0],
    [0,0,1,1],
    [1,0,1,0],
    [0,1,0,1]
])

#Definindo o numero de cluster/grupos
num_cluster = 2

#Inicializar o modelo
kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=10)

#Treinando o modelo
kmeans.fit(filmes_assitidos)

#Classificando os usuários

grupos_indice = kmeans.predict(filmes_assitidos)

#Exibir os dados
print("Usuário pertence ao seguinte grupo:")

for i, cluster in enumerate(grupos_indice):
  print(f"Usuário {i+1} pertence ao grupo {cluster+1}")

print("\nFilmes assitidos")
for i in range(len(filmes_assitidos)):
  assitidos = np.where(filmes_assitidos[i] == 1)[0] + 1
  print(f"Usuário {i+1} assitidos aos filmes: {assitidos}")

#Função que recomenda filmes
def recomendar_filmes(filmes, filmes_assitidos, grupos_indice):

  filmes = np.array(filmes)

  #Encontrar o grupo do usário com base em seu vetor de filmes assitidos
  usuario_id = len(filmes_assitidos)
  grupo_usuario = kmeans.predict([filmes])[0]

  #Encontrar todos od usuários no mesmo grupo
  usuarios_no_mesmo_grupo = [i for i in range(len(grupos_indice))if grupos_indice[i] == grupo_usuario]

  #Filmes assitidos pelos usuários no mesmo grupo
  filmes_recomendados = set ()
  for usuario in usuarios_no_mesmo_grupo:
    filmes_assitidos_usuario = np.where(filmes_assitidos[usuario] == 1)[0]
    filmes_recomendados.update(filmes_assitidos_usuario)

  #Remover filmes que o usuário já assistiu
  filmes_recomendados = filmes_recomendados - set(np.where(filmes == 1)[0])

  #Ajustar os indeces dos filmes recomendados (de volta para 1-based)
  filmes_recomendados = [filmes + 1 for filme in filmes_recomendados]

  return sorted(filmes_recomendados)

#Exemplo de uso da função recomendar_filmes
filmes_assitidos_usuario = [1, 0, 1, 0] #vetor de filmes
#Assitidos(por exemplo, assitiu aos filmes 1 e 3)
filmes_recomendados = recomendar_filmes(filmes_assitidos_usuario, filmes_assitidos, grupos_indice)

print(f"\nFilmes recomendados: {filmes_recomendados}")