import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Exemplo de dados - você deve substituir isso pelos seus dados reais
data = {
    'mes': np.arange(1, 13).tolist(),  # Meses de 1 a 12
    'vendas': [200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500],  # Vendas em cada mês
    'promocao': [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],  # 1 se houve promoção, 0 caso contrário
    'sazonalidade': [1, 1.1, 1.2, 1.3, 1.5, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1],  # Fator sazonal
}
print(data.head())
# Criando DataFrame
df = pd.DataFrame(data)

# Criação de variáveis para regressão
# Ajustando a sazonalidade
df['vendas_sazonais'] = df['vendas'] * df['sazonalidade']

# Selecionando variáveis independentes e dependentes
X = df[['mes', 'promocao', 'sazonalidade']]
y = df['vendas']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação do modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Erro Médio Quadrático (MSE):", mse)
print("Coeficiente de Determinação (R^2):", r2)

# Visualizar resultados
plt.scatter(y_test, y_pred)
plt.xlabel('Valores reais')
plt.ylabel('Previsões')
plt.title('Valores reais vs Previsões')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)  # Linha diagonal
plt.show()