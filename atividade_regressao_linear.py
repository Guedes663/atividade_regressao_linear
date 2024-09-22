import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

caminho_csv = './dados/Estudo_vs_Nota_Dataset__Varied_.csv'
df = pd.read_csv(caminho_csv)

# print(df.head())
# print(df.describe())

# plt.scatter(df['HorasEstudo'], df['NotaProva'])
# plt.xlabel('Horas de Estudo')
# plt.ylabel('Nota da Prova')
# plt.title('Nota da Prova vs Horas de Estudo')
# plt.show()

df['HorasEstudo'] = df['HorasEstudo'].abs()
df['NotaProva'] = df['NotaProva'].abs()

condicao_x = (df['HorasEstudo'] >= 0)
print('Valores inválidos na variável independente (X):\n', df[~condicao_x])

condicao_y = (df['NotaProva'] >= 0) & (df['NotaProva'] <= 100)
print('Valores inválidos na variável dependente (Y):\n', df[~condicao_y])

df_limpo = df[condicao_x & condicao_y]

if not df_limpo.empty:
    x = df_limpo[['HorasEstudo']]
    y = df_limpo['NotaProva']

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

    modelo = LinearRegression()

    modelo.fit(x_treino, y_treino)

    previsao_y = modelo.predict(x_teste)

    mse = mean_squared_error(y_teste, previsao_y)
    r2 = r2_score(y_teste, previsao_y)

    print(f'MSE: {mse}')
    print(f'R²: {r2}')

    plt.scatter(x_teste, y_teste, color='blue', label='Dados reais')
    plt.plot(x_teste, previsao_y, color='red', label='Linha de regressão')
    plt.title('Regressão Linear')
    plt.xlabel('Horas de Estudo')
    plt.ylabel('Nota da Prova')
    plt.legend()
    plt.show()
else:
    print('Não há dados suficientes após a limpeza.')
