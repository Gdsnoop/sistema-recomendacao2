#Importando as bibliotecas
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

#1-Especificar o caminho do arquivo CSV
caminho_arquivo = '/content/filmes_100_usuarios.csv'

#2-Ler o CSV
df = pd.read_csv(caminho_arquivo)

#Verificar se foi lido corretamente exibindo eles
print(df.head())

#Matriz com os filmes vistos
#Convertendo o DataFrame para uma matrix numpy
filmes_assistidos = df.drop(columns=["Unnamed: 0"]).values

#Definindo o numero de cluster/grupos
num_cluster = 2

#Inicializar o modelo
kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=10)

#Treinando o modelo
kmeans.fit(filmes_assistidos)

#Classificando os usuários

grupos_indice = kmeans.predict(filmes_assistidos)

#Exibir os dados
print("Usuário pertence ao seguinte grupo:")

for i, cluster in enumerate(grupos_indice):
  print(f"Usuário {i+1} pertence ao grupo {cluster+1}")

print("\nFilmes assitidos")
for i in range(len(filmes_assistidos)):
  assitidos = np.where(filmes_assistidos[i] == 1)[0] + 1
  print(f"Usuário {i+1} assistidos aos filmes: {assitidos}")

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
  filmes_recomendados = [filme + 1 for filme in filmes_recomendados]

  return sorted(filmes_recomendados)

#Exemplo de uso da função recomendar_filmes
filmes_assitidos_usuario = [1, 0, 0, 1, 1, 1, 1, 0, 0, 1] #vetor de filmes
#Assitidos(por exemplo, assitiu aos filmes 1 e 3)
filmes_recomendados = recomendar_filmes(filmes_assitidos_usuario, filmes_assistidos, grupos_indice)

print(f"\nFilmes recomendados: {filmes_recomendados}")