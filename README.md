# Projeto Final: Análise de Vendas E-commerce

**Autor:** [Seu Nome Completo Aqui]
**Curso:** Data Science

---

## 1. Objetivo do Projeto

Realizar uma análise detalhada sobre um conjunto de dados de vendas de uma empresa de e-commerce para identificar padrões, extrair insights e desenvolver um modelo de machine learning capaz de prever o faturamento futuro.

---

## 2. Metodologia

O projeto foi estruturado nas seguintes etapas:

1.  **Preparação dos Dados:** Criação de um dataset sintético e realização de limpeza e formatação.
2.  **Análise Exploratória (EDA):** Investigação dos dados através de estatísticas descritivas e visualizações gráficas.
3.  **Teste de Hipótese:** Aplicação do teste ANOVA para verificar a relevância estatística da variável "Categoria".
4.  **Modelagem Preditiva:** Desenvolvimento de um modelo de Regressão Linear para prever o `Valor_Total` das vendas.

---

## 3. Análise Exploratória: Principais Descobertas



* Observa-se uma clara sazonalidade, com picos de vendas nos meses de Novembro e Dezembro, consistentes com os períodos de Black Friday e Natal.



* A categoria "Eletrônicos" domina o faturamento, devido ao seu alto valor agregado por item.

---

## 4. Teste de Hipótese (ANOVA)

**Questão:** A categoria de um produto tem um impacto estatisticamente significativo no valor total da venda?

* **Hipótese Nula (H0):** As médias de vendas são iguais para todas as categorias.
* **Resultado:** O teste ANOVA apresentou um **p-valor < 0.0001**.
* **Conclusão:** Rejeitamos a hipótese nula. Existe uma diferença estatisticamente significativa no valor das vendas entre as diferentes categorias de produtos, confirmando que esta é uma variável importante para o modelo.

---

## 5. Modelo de Machine Learning: Previsão de Vendas

Foi treinado um modelo de **Regressão Linear** para prever o valor das vendas.

**Resultados da Avaliação:**

| Métrica                 | Valor    |
| ----------------------- | -------- |
| **R-squared (R²)** | 0.9416   |
| **Mean Absolute Error** | 1081.79  |
| **Mean Squared Error** | 2505676.43 |

* O **R² de 0.9416** indica que o modelo consegue explicar aproximadamente 94% da variabilidade no valor das vendas, demonstrando um ótimo ajuste aos dados.

---

## 6. Conclusões e Próximos Passos

**Conclusões:**
* As vendas possuem forte influência da sazonalidade e da categoria do produto.
* A categoria "Eletrônicos" é o principal motor de receita da empresa.
* O modelo de Regressão Linear se mostrou eficaz para prever o faturamento com alta precisão.

**Próximos Passos:**
* Incorporar dados externos, como investimentos em marketing ou feriados regionais.
* Testar modelos mais robustos (ex: Gradient Boosting, Redes Neurais).
* Desenvolver um modelo específico para prever a quantidade de unidades vendidas.

---

## Obrigado!