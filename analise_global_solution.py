"""
Script de Análise de Dados para o Projeto Global Solution - 1º Semestre 2025

Este script realiza uma análise exploratória e constrói um modelo preditivo
utilizando a base de dados de risco de enchente da Índia.

Autor: [SEU NOME COMPLETO AQUI]
R.A.: [SEU R.A. AQUI]
Data de Criação: 06/06/2025
"""

# ==============================================================================
# 0. IMPORTAÇÃO DE DEPENDÊNCIAS
# ==============================================================================
print("Inicializando o script: importando dependências...")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
print("Dependências carregadas com sucesso.\n")

# ==============================================================================
# 1. CARREGAMENTO E INSPEÇÃO INICIAL DOS DADOS
# ==============================================================================
print("--- [ETAPA 1: Carregamento e Inspeção dos Dados] ---")
DATA_FILE_PATH = 'flood_risk_dataset_india.csv' 

try:
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Dataset '{DATA_FILE_PATH}' carregado com sucesso.")
    print("\nAmostra do DataFrame:")
    print(df.head())
    print("\nEstrutura do DataFrame (colunas):", df.columns.tolist())
except FileNotFoundError:
    print(f"\n[ERRO CRÍTICO] O arquivo de dados '{DATA_FILE_PATH}' não foi encontrado.")
    print("O script será encerrado. Verifique se o arquivo está na mesma pasta.")
    exit()

# ==============================================================================
# 2. ANÁLISE DE DISTRIBUIÇÃO DE FREQUÊNCIA
# ==============================================================================
print("\n\n--- [ETAPA 2: Análise de Frequência] ---")

# 2.a) Variável Quantitativa Discreta
print("\n2.a) Frequência da variável discreta 'Historical Floods'")
freq_discreta = df['Historical Floods'].value_counts().sort_index()
print(pd.DataFrame({'Frequência': freq_discreta}))

# 2.b) Variável Quantitativa Contínua
print("\n2.b) Frequência da variável contínua 'Rainfall (mm)'")
classes_continuas = pd.cut(df['Rainfall (mm)'], bins=8, right=False)
freq_continua = classes_continuas.value_counts().sort_index()
print(pd.DataFrame({'Frequência': freq_continua}))

# ==============================================================================
# 3. VISUALIZAÇÃO DE DADOS
# ==============================================================================
print("\n\n--- [ETAPA 3: Geração de Visualizações] ---")
print("A execução do script será pausada para exibir cada gráfico.")
sns.set_theme(style="whitegrid", palette="viridis")

# Gráfico 1: Barras para a variável discreta
plt.figure(figsize=(12, 7))
ax1 = sns.countplot(x='Historical Floods', data=df, hue='Historical Floods', legend=False) 
ax1.set_title('Gráfico 1: Frequência de Enchentes Históricas', fontsize=16, pad=20)
ax1.set_xlabel('Número de Enchentes Históricas Registradas', fontsize=12)
ax1.set_ylabel('Contagem (Frequência)', fontsize=12)
plt.show()

# Gráfico 2: Histograma para a variável contínua
plt.figure(figsize=(12, 7))
ax2 = sns.histplot(df['Rainfall (mm)'], bins=20, kde=True, color='dodgerblue')
ax2.set_title('Gráfico 2: Distribuição de Frequência da Precipitação', fontsize=16, pad=20)
ax2.set_xlabel('Precipitação (mm)', fontsize=12)
ax2.set_ylabel('Frequência', fontsize=12)
plt.show()

# ==============================================================================
# 4. ANÁLISE ESTATÍSTICA DESCRITIVA
# ==============================================================================
print("\n\n--- [ETAPA 4: Análise Estatística Descritiva] ---")
variavel_analise = df['Rainfall (mm)']
print(f"Analisando a variável: '{variavel_analise.name}'")

media = variavel_analise.mean()
mediana = variavel_analise.median()
moda = variavel_analise.mode()[0]
minimo, maximo = variavel_analise.min(), variavel_analise.max()
amplitude = maximo - minimo
variancia = variavel_analise.var()
desvio_padrao = variavel_analise.std()
coef_variacao = (desvio_padrao / media) * 100 if media != 0 else 0
q1, q2, q3 = variavel_analise.quantile([0.25, 0.50, 0.75])

print("-" * 60)
print(f"{'Medida':<25} | {'Valor':>20}")
print("-" * 60)
print(f"{'Média':<25} | {media:20,.2f}")
print(f"{'Mediana':<25} | {mediana:20,.2f}")
print(f"{'Moda':<25} | {moda:20,.2f}")
print("-" * 60)
print(f"{'Mínimo':<25} | {minimo:20,.2f}")
print(f"{'Máximo':<25} | {maximo:20,.2f}")
print(f"{'Amplitude Total':<25} | {amplitude:20,.2f}")
print(f"{'Variância':<25} | {variancia:20,.2f}")
print(f"{'Desvio Padrão':<25} | {desvio_padrao:20,.2f}")
print(f"{'Coeficiente de Variação (%)':<25} | {coef_variacao:19,.2f}%")
print("-" * 60)
print(f"{'1º Quartil (Q1)':<25} | {q1:20,.2f}")
print(f"{'2º Quartil (Q2 - Mediana)':<25} | {q2:20,.2f}")
print(f"{'3º Quartil (Q3)':<25} | {q3:20,.2f}")
print("-" * 60)

# ==============================================================================
# 5. MODELAGEM PREDITIVA: REGRESSÃO LINEAR SIMPLES
# ==============================================================================
print("\n\n--- [ETAPA 5: Construção do Modelo Preditivo] ---")
print("Objetivo: Prever 'Flood Occurred' a partir de 'Rainfall (mm)'.")

X = df[['Rainfall (mm)']]
y = df['Flood Occurred']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nDados divididos: {len(X_train)} amostras de treino, {len(X_test)} amostras de teste.")

modelo_regressao = LinearRegression()
modelo_regressao.fit(X_train, y_train)
print("Modelo de Regressão Linear treinado.")

coeficiente_angular = modelo_regressao.coef_[0]
intercepto = modelo_regressao.intercept_
print(f"\nCoeficiente Angular (β1): {coeficiente_angular:.6f}")
print(f"Intercepto (β0): {intercepto:.4f}")

score_r2 = modelo_regressao.score(X_test, y_test)
print(f"\nScore R² (Teste): {score_r2:.4f} ({score_r2:.2%})")

# ==============================================================================
# 6. EXPORTAÇÃO DOS DADOS PARA ENTREGA
# ==============================================================================
OUTPUT_FILENAME = "dados_analise_enchente.xlsx"
try:
    df.to_excel(OUTPUT_FILENAME, index=False)
    print(f"\n\n--- [ETAPA 6: Exportação de Dados] ---")
    print(f"DataFrame completo exportado com sucesso para '{OUTPUT_FILENAME}'")
except Exception as e:
    print(f"\n[ERRO] Falha ao exportar dados para Excel: {e}")

print("\nExecução do script finalizada com sucesso.")