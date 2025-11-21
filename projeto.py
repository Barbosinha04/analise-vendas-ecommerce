import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os


def criar_e_preparar_dados():
    np.random.seed(42)
    datas = pd.to_datetime(pd.date_range(start='2023-01-01', end='2024-12-31', freq='D'))
    num_registros = len(datas)

    categorias = ['Eletrônicos', 'Vestuário', 'Casa e Jardim', 'Livros']
    dados = {
        'Data': np.random.choice(datas, size=num_registros * 2, replace=True),
        'Categoria': np.random.choice(categorias, size=num_registros * 2, replace=True)
    }
    df = pd.DataFrame(dados)

    preco_base = {'Eletrônicos': 1200, 'Vestuário': 150, 'Casa e Jardim': 250, 'Livros': 50}
    df['Preco_Unitario'] = df['Categoria'].map(preco_base)
    df['Preco_Unitario'] = df['Preco_Unitario'] * (1 + np.random.randn(len(df)) * 0.1)

    df['Unidades_Vendidas'] = np.random.randint(1, 15, size=len(df))
    df.loc[df['Data'].dt.month == 12, 'Unidades_Vendidas'] *= 2
    df.loc[df['Data'].dt.month == 11, 'Unidades_Vendidas'] = (
                df.loc[df['Data'].dt.month == 11, 'Unidades_Vendidas'] * 1.5).astype(int)

    df['Valor_Total'] = df['Preco_Unitario'] * df['Unidades_Vendidas']
    df['Data'] = pd.to_datetime(df['Data'])

    df.dropna(inplace=True)
    df = df.drop_duplicates()
    df = df.sort_values('Data').reset_index(drop=True)

    return df


def analise_exploratoria(df):
    if not os.path.exists('graficos'):
        os.makedirs('graficos')

    print("--- Informações Gerais do DataFrame ---")
    df.info()
    print("\n--- Estatísticas Descritivas ---")
    print(df.describe())

    df_mensal = df.set_index('Data').resample('M')['Valor_Total'].sum()

    plt.style.use('seaborn-v0_8-grid')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_mensal.index, df_mensal.values, marker='o', linestyle='-')
    ax.set_title('Evolução das Vendas Mensais (2023-2024)', fontsize=16)
    ax.set_xlabel('Mês', fontsize=12)
    ax.set_ylabel('Valor Total Vendido', fontsize=12)
    plt.tight_layout()
    plt.savefig('graficos/vendas_mensais.png')
    plt.close()

    vendas_categoria = df.groupby('Categoria')['Valor_Total'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=vendas_categoria.index, y=vendas_categoria.values, palette='viridis', ax=ax)
    ax.set_title('Total de Vendas por Categoria', fontsize=16)
    ax.set_xlabel('Categoria', fontsize=12)
    ax.set_ylabel('Valor Total Vendido', fontsize=12)
    plt.tight_layout()
    plt.savefig('graficos/vendas_por_categoria.png')
    plt.close()


def teste_de_hipotese(df):
    print("\n--- Teste de Hipótese (ANOVA) ---")
    print("H0: Não há diferença significativa nas médias de vendas entre as categorias.")
    print("H1: Existe diferença significativa na média de vendas de pelo menos uma categoria.")

    vendas_eletronicos = df[df['Categoria'] == 'Eletrônicos']['Valor_Total']
    vendas_vestuario = df[df['Categoria'] == 'Vestuário']['Valor_Total']
    vendas_casa = df[df['Categoria'] == 'Casa e Jardim']['Valor_Total']
    vendas_livros = df[df['Categoria'] == 'Livros']['Valor_Total']

    f_stat, p_valor = f_oneway(vendas_eletronicos, vendas_vestuario, vendas_casa, vendas_livros)

    print(f"Estatística F: {f_stat:.4f}")
    print(f"P-valor: {p_valor:.4f}")

    alpha = 0.05
    if p_valor < alpha:
        print(
            "Resultado: Rejeitamos a hipótese nula. As médias de vendas entre as categorias são significativamente diferentes.")
    else:
        print("Resultado: Falhamos em rejeitar a hipótese nula. Não há evidências de diferença significativa.")


def modelo_machine_learning(df):
    print("\n--- Modelagem de Machine Learning ---")

    df_modelo = df.copy()
    df_modelo['Mes'] = df_modelo['Data'].dt.month
    df_modelo['Ano'] = df_modelo['Data'].dt.year
    df_modelo['Dia_da_Semana'] = df_modelo['Data'].dt.dayofweek

    df_modelo = pd.get_dummies(df_modelo, columns=['Categoria'], drop_first=True)

    features = [
        'Mes', 'Ano', 'Dia_da_Semana',
        'Categoria_Eletrônicos', 'Categoria_Livros', 'Categoria_Vestuário'
    ]
    target = 'Valor_Total'

    X = df_modelo[features]
    y = df_modelo[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Avaliando o modelo de Regressão Linear:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")


if __name__ == '__main__':
    dataframe_vendas = criar_e_preparar_dados()
    analise_exploratoria(dataframe_vendas)
    teste_de_hipotese(dataframe_vendas)
    modelo_machine_learning(dataframe_vendas)
    print("\nAnálise concluída. Gráficos salvos na pasta 'graficos'.")