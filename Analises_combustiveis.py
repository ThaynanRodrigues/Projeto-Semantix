import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
import numpy as np

# ================================
# 1. CONFIGURAÇÕES INICIAIS
# ================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')
os.makedirs(PLOTS_PATH, exist_ok=True)

# ================================
# 2. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ================================

all_files = glob.glob(os.path.join(BASE_PATH, "*.csv"))
df_list = []

for file in all_files:
    df = pd.read_csv(file, sep=';', encoding='utf-8', dtype=str)
    df['Valor de Venda'] = pd.to_numeric(
        df['Valor de Venda'].str.replace(',', '.'), errors='coerce')
    df['Valor de Compra'] = pd.to_numeric(
        df['Valor de Compra'].str.replace(',', '.'), errors='coerce')
    df['Data da Coleta'] = pd.to_datetime(
        df['Data da Coleta'], format="%d/%m/%Y", errors='coerce')
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
full_df.dropna(
    subset=['Valor de Venda', 'Data da Coleta', 'Produto'], inplace=True)

# ================================
# 3. TENDÊNCIA DE PREÇOS POR PRODUTO
# ================================

avg_price_time = full_df.groupby(['Data da Coleta', 'Produto'])[
    'Valor de Venda'].mean().unstack()
avg_price_time.plot(
    figsize=(16, 6), title='Evolução do Preço Médio por Produto')
plt.ylabel('Preço Médio (R$)')
plt.xlabel('Data da Coleta')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'tendencia_precos_por_produto.png'))
plt.close()

# ================================
# 4. MARGEM BRUTA
# ================================

full_df['Margem Bruta'] = full_df['Valor de Venda'] - \
    full_df['Valor de Compra']
margem_por_produto = full_df.groupby(
    'Produto')['Margem Bruta'].mean().sort_values(ascending=False)

print("\n🔍 Margem média por produto:")
print(margem_por_produto)

# ================================
# 5. RANKING DE PREÇOS POR MUNICÍPIO
# ================================

ranking = (
    full_df.groupby(['Estado - Sigla', 'Municipio'])['Valor de Venda']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\n🏆 Top 10 municípios com maiores preços médios:")
print(ranking)

# ================================
# 6. COMPARATIVO ENTRE BANDEIRAS
# ================================

bandeira_comparativo = full_df.groupby(['Bandeira', 'Produto'])[
    'Valor de Venda'].mean().unstack()
print("\n📊 Preço médio por produto e bandeira:")
print(bandeira_comparativo)

bandeira_comparativo.plot(kind='bar', figsize=(
    14, 6), title='Preço Médio por Produto e Bandeira')
plt.ylabel('Preço Médio (R$)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'comparativo_bandeiras.png'))
plt.close()

# ================================
# 7. DETECÇÃO DE OUTLIERS (Gasolina)
# ================================

gas_df = full_df[full_df['Produto'] == 'GASOLINA'].copy()
gas_df['Z_score'] = stats.zscore(gas_df['Valor de Venda'].dropna())
outliers = gas_df[gas_df['Z_score'].abs() > 3]

print(f"\n⚠️ Total de outliers encontrados para Gasolina: {len(outliers)}")
print(outliers[['Data da Coleta', 'Municipio',
      'Valor de Venda', 'Z_score']].head())

plt.figure(figsize=(12, 5))
sns.histplot(gas_df['Valor de Venda'], kde=True, bins=30)
plt.axvline(outliers['Valor de Venda'].min(), color='red',
            linestyle='--', label='Outlier Min')
plt.axvline(outliers['Valor de Venda'].max(), color='red',
            linestyle='--', label='Outlier Max')
plt.title('Distribuição do Preço da Gasolina com Outliers')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'outliers_gasolina.png'))
plt.close()

# ================================
# 8. PREVISÃO COM REGRESSÃO LINEAR (Gasolina)
# ================================

gasolina = full_df[full_df['Produto'] == 'GASOLINA'].dropna(
    subset=['Valor de Venda', 'Data da Coleta'])
gasolina.sort_values('Data da Coleta', inplace=True)

X = gasolina['Data da Coleta'].map(
    pd.Timestamp.toordinal).values.reshape(-1, 1)
y = gasolina['Valor de Venda'].values

model = LinearRegression()
model.fit(X, y)
gasolina['Previsao'] = model.predict(X)

plt.figure(figsize=(14, 6))
plt.plot(gasolina['Data da Coleta'],
         gasolina['Valor de Venda'], label='Valor Real')
plt.plot(gasolina['Data da Coleta'], gasolina['Previsao'],
         label='Previsão Linear', linestyle='--')
plt.title('Tendência Linear do Preço da Gasolina')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'previsao_linear_gasolina.png'))
plt.close()

# ================================
# 9. MÉDIA MENSAL POR PRODUTO
# ================================

full_df['AnoMes'] = full_df['Data da Coleta'].dt.to_period('M')
media_mensal = full_df.groupby(['AnoMes', 'Produto'])[
    'Valor de Venda'].mean().unstack()

# Exibe os dados no terminal
print("\n📆 Média mensal por produto:")
print(media_mensal.round(2))

media_mensal.plot(figsize=(16, 6), title='Preço Médio Mensal por Produto')
plt.ylabel('Preço (R$)')
plt.xlabel('Ano-Mês')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'preco_mensal_produto.png'))
plt.close()

# ================================
# 10. MÉDIA SEMESTRAL POR PRODUTO
# ================================

full_df['Ano'] = full_df['Data da Coleta'].dt.year
full_df['Semestre'] = full_df['Data da Coleta'].dt.month.apply(
    lambda m: 1 if m <= 6 else 2)
full_df['Ano_Semestre'] = full_df['Ano'].astype(
    str) + '-' + full_df['Semestre'].astype(str)

media_semestral = full_df.groupby(['Ano_Semestre', 'Produto'])[
    'Valor de Venda'].mean().unstack()

print("\n🌓 Média semestral por produto:")
print(media_semestral.round(2))

media_semestral.plot(
    figsize=(16, 6), title='Preço Médio Semestral por Produto')
plt.ylabel('Preço Médio (R$)')
plt.xlabel('Ano-Semestre')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'preco_semestral_produto.png'))
plt.close()

print("📘 Média semestral gerada e salva em 'preco_semestral_produto.png'")

# ================================
# 11. MÉDIA ANUAL POR PRODUTO
# ================================

media_anual = full_df.groupby(['Ano', 'Produto'])[
    'Valor de Venda'].mean().unstack()

print("\n📅 Média anual por produto:")
print(media_anual.round(2))

media_anual.plot(figsize=(16, 6), title='Preço Médio Anual por Produto')
plt.ylabel('Preço Médio (R$)')
plt.xlabel('Ano')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'preco_anual_produto.png'))
plt.close()

print("📕 Média anual gerada e salva em 'preco_anual_produto.png'")

# ================================
# FIM
# ================================

print("\n✅ Análise concluída. Todos os gráficos foram salvos na pasta 'plots/'.")
