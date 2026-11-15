# 🎓 Desenvolvimento de um Sistema Inteligente para Classificação de Artefatos em Mamografias utilizando Transformada Wavelet Integrada à Pipeline da CNN e do SWIN Transformer

![Curso](https://img.shields.io/badge/Curso-Engenharia_de_Telecomunicações-blue)
![Ano](https://img.shields.io/badge/Ano-2026-yellow)

## Sobre o Trabalho
<p align="justify">
O câncer de mama está entre as principais causas de incidência e mortalidade em mulheres no Brasil e no mundo, e sua identificação em estágios iniciais é fundamental para aumentar as taxas de sobrevivência. Este trabalho investiga o uso de modelos de aprendizado profundo para a classificação de imagens mamográficas, comparando duas arquiteturas pré-treinadas: ResNet50, baseada em redes neurais convolucionais (CNN), e Swin Transformer Tiny (Swin-T), por meio de métodos de adaptação dos modelos. A abordagem proposta consiste na substituição das camadas convencionais de subamostragem por uma Transformada Wavelet, buscando preservar informações relevantes das imagens durante a extração de características. Os experimentos utilizam as bases públicas Mini-MIAS e Mini-DDSM, com etapas de pré-processamento, aumento de dados e avaliação por métricas como acurácia, sensibilidade, especificidade, F1-score e AUC-ROC. Além da avaliação interna, na qual treinamento e teste são realizados na mesma base de dados, é realizada uma avaliação externa, utilizando uma base para treinamento e outra para teste, a fim de investigar a capacidade de generalização dos modelos diante de diferentes distribuições de dados. Busca-se identificar uma abordagem que combine bom desempenho e capacidade de generalização, mantendo resultados consistentes em diferentes bases de dados e contribuindo para o desenvolvimento de ferramentas de apoio à análise de exames mamográficos.</p>

---

## Estrutura do Repositório

```text
├── 📂 backend_mamografia/   # Aplicação web experimental 
├── 📂 dados_mamografia/     # Imagens, datasets, exames anotados e pré-processamento
├── 📂 documento/            # Texto final do TCC em Typst
├── 📂 slides/               # Apresentações das bancas e qualificações 
├── 📂 experimentos/         # Código dos modelos de aprendizado (CNN, SWIN e Dual)
└── README.md                # Documentação geral do repositório
