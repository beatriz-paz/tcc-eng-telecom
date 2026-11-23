# 🎓 Desenvolvimento de um Sistema Inteligente para Classificação de Artefatos em Mamografias utilizando Transformada Wavelet Integrada à Pipeline da CNN e do SWIN Transformer

![Curso](https://img.shields.io/badge/Curso-Engenharia_de_Telecomunicações-blue)
![Ano](https://img.shields.io/badge/Ano-2026-yellow)

## Resumo
<p align="justify">
O câncer de mama está entre as principais causas de incidência e mortalidade em mulheres no Brasil e no mundo, e sua identificação em estágios iniciais é fundamental para aumentar as taxas de sobrevivência. Este trabalho investiga o uso de modelos de aprendizado profundo para a classificação de imagens mamográficas, comparando duas arquiteturas pré-treinadas: ResNet50, baseada em redes neurais convolucionais (CNN), e Swin Transformer Tiny (Swin-T), por meio de métodos de adaptação dos modelos. A abordagem proposta consiste na substituição das camadas convencionais de subamostragem por uma Transformada Wavelet, buscando preservar informações relevantes das imagens durante a extração de características. Os experimentos utilizam as bases públicas Mini-MIAS e Mini-DDSM, com avaliação por métricas como AUC-ROC, sensibilidade e especificidade, em validação interna (treinamento e teste na mesma base) e externa (treinamento em uma base e teste na outra), a fim de investigar a capacidade de generalização dos modelos. Foram avaliados 27 modelos em cinco grupos experimentais. O recorte da região de interesse (ROI) elevou a AUC no Mini-MIAS de 48,2% para 78,6% na CNN e de 52,8% para 85,7% no Swin-T. A substituição da subamostragem pela Wavelet, fixa ou treinável, não superou os modelos de referência. O Swin-T com SSF apresentou o melhor equilíbrio entre desempenho e generalização, com AUC externa de 80,2% (Mini-MIAS → Mini-DDSM) e 91,4% (Mini-DDSM → Mini-MIAS) e as menores quedas em relação à validação interna (2,4 e 3,8 p.p.). A transferência entre bases foi assimétrica, sendo mais favorável quando o treinamento é feito na base maior. Conclui-se que o recorte da ROI e a estratégia de adaptação influenciaram o desempenho e a generalização mais do que a troca da subamostragem por Wavelet, e que a avaliação externa é indispensável, pois o desempenho interno isoladamente não reflete o comportamento dos modelos em outras bases.</p>

---

## Estrutura do Repositório

```text
├── 📂 backend_mamografia/   # Aplicação web experimental 
├── 📂 dados_mamografia/     # Imagens, datasets, exames anotados e pré-processamento
├── 📂 documento/            # Texto final do TCC em Typst
├── 📂 experimentos/         # Código dos modelos de aprendizado (CNN, SWIN e Dual)
├── 📂 slides/               # Apresentações do TCC para a banca  
└── README.md                
