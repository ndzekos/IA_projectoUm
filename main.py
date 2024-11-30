import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Carregando  o dataset local do projecto
file_path = 'supermarket.csv'  # Atualize o caminho conforme necessário
df = pd.read_csv(file_path)
# Visualizando as primeiras linhas do DataFrame
#print(df.head())


# EXPLORAÇÃO DOS DADOS.
#PARA VISUALIZAR A ESTRURURA DO DATAFRAME USANOS  O df.info().
# PARA EXIBIR A ESTATISTICA DESCRITIVA DAS COLUNAS NUMERICAS COM MEDIA, USAMOS O df.describe()

# Informações sobre o DataFrame
# print(df.info())

# Estatísticas descritivas
# print(df.describe())

# LIMPEZA DE DADOS
# Remover dados em falta
df.dropna(inplace=True)

# Remover duplicatas com base no Invoice ID
df.drop_duplicates(subset='Invoice ID', inplace=True)

# Remover dados duplicados
df.drop_duplicates(inplace=True)

# CONVERSÃO DE DATA PARA O PADRÃO NORMAL PT-PT

# Converter a coluna 'Date' para o tipo datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Criar a coluna com o formato '05/01/2019' (pt-BR)
df['Date_Venda'] = df['Date'].dt.strftime('%d/%m/%Y')

# Criar a coluna com o formato '01-05-2019'
df['Date'] = df['Date'].dt.strftime('%m-%d-%Y')

# Exibir o DataFrame resultante
print(df.head())

# MODELAGEM.
# Comparando o total de vendas para cada uma da tres filiais. Com o objectivo a identificar qual dos flilais teve o maior
# volume de vendas no periodo analisado.
print("1. Comparando o valor total de vendas por empresa.")
valor_total_vendas_filial = df.groupby('Branch')['Total'].sum()
print(valor_total_vendas_filial)

# Vertificacao do Equilibrio em valor de vendas pelas tres filiais.
print("2. Equilibrio de venda para os tres filiais em forma de Media de venda dos clientes")
media_vendas_cliente = df.groupby('Customer type')['Total'].mean()
print(media_vendas_cliente)

# Observando a media de vendas por tipo de cliente.
# sns.countplot(x='Payment', data=df)
# plt.title('Distribuição de Métodos de Pagamento')
# plt.show()

# Para a distribuição por métodos de pagamentos é possível notar um equilíbrio bem grande
# também, com cartão de crédito sendo o método com menor quantidade de
# pedidos, mas uma difernça bem pequena.
valor_venda_media_produto = df.groupby('Product line')['Total'].mean().sort_values(ascending=False)
print(valor_venda_media_produto)


# Equilibrio para a media de valor de vendas por categoria de produtos.
print("3.Equilibrio para a media de valor de vendas por categoria de produtos")
lucro_medio_produto = df.groupby('Product line')['gross income'].mean().sort_values(ascending=False)
print(lucro_medio_produto)

#Analisando as medias de lucro.
# Converter a coluna 'Date' para datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%Y')

# Definir a coluna 'Date' como índice para facilitar o agrupamento semanal
df.set_index('Date', inplace=True)

# Acima Definimos o indice usando a coluna Date para que a visualizacao dos dados em um grafico temporario seja claro.


# Analisando o comportamento dass vendas de forma temporal, em consideracao da distribuicao semanal.
# Agrupar a quantidade de vendas por semana
qtd_vendas_por_semana = df['Total'].resample('W').count()

# Plotar o total de vendas semanais
qtd_vendas_por_semana.plot(figsize=(10, 5), title="Total de Vendas Semanais")
# X_train=plt.xlabel("Data")
# y_train=plt.ylabel("Vendas Totais")
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
plt.xlabel("Data")
plt.ylabel("Vendas Totais")
model = LinearRegression()


plt.show()
# No grafico gerado acima, e possivel anotar o pico de vendas entre os dias 20 a 27 de janeiro para o dataset em causa.


# Agrupando  as vendas totais por semana
valor_vendas_por_semana = df['Total'].resample('W').sum()

# Agrupar a quantidade de vendas por semana
qtd_vendas_por_semana = df['Total'].resample('W').count()

# Plotar o total de vendas semanais
valor_vendas_por_semana.plot(figsize=(10, 5), title="Valor Total em Vendas Semanais")
plt.xlabel("Data")
plt.ylabel("Valor Total de Vendas")
plt.show()


# Ao observar este segundo gráfico, que exibe o valor das vendas em períodos regulares, percebe-se uma
# similaridade com o gráfico de quantidade de vendas. Esse padrão é esperado, reforçando que
# os clientes tendem a concentrar suas compras nas primeiras semanas do mês, elevando os lucros
# dos supermercados nesses períodos.

