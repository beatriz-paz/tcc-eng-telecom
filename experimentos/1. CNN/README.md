# ResNet50 para classificação de mamografias

## Resumo

Esses notebooks treinam uma **ResNet50** (pré-treinada no ImageNet-1k) para classificar mamografias como **Normal** ou **Anormal** (benigno ou maligno). O foco não é apenas medir o desempenho dentro de um banco de dados, mas também verificar quanto esse desempenho se mantém quando o modelo é avaliado em outro banco. Assim, é possível investigar se o modelo aprendeu características que podem ser transferidas entre datasets ou se depende muito das características do conjunto usado no treinamento.

Para isso, cada notebook utiliza dois datasets públicos (**mini-MIAS** e **Mini-DDSM**) e realiza dois experimentos:

1. **Teste interno:** treinamento e teste no mesmo dataset, por validação cruzada.
2. **Teste externo:** treinamento em um dataset e avaliação no outro, nos dois sentidos.

---

## 1. Generalização

Modelos de deep learning para mamografia podem apresentar bons resultados quando são treinados e avaliados em imagens de um mesmo dataset, mas nem sempre mantêm esse desempenho quando recebem imagens de outra origem. Essa diferença entre os conjuntos é conhecida como **domain shift** (mudança de domínio).

Este notebook investiga esse efeito para uma arquitetura CNN, a ResNet50, e busca responder a três perguntas:

- Qual é o desempenho do modelo quando treino e teste pertencem ao mesmo dataset?
- Quanto o AUC cai quando o modelo é avaliado em outro dataset?
- O limiar de decisão escolhido durante o treinamento continua adequado no dataset de destino?

## 2. Dados

| Dataset | Imagens utilizadas | Anormais | Observação |
|---|---:|---:|---|
| **mini-MIAS** | 318 | 111 (~35 %) | Dataset menor; as duas mamas de uma mesma paciente são agrupadas na divisão dos dados. |
| **Mini-DDSM** (Complete, JPEG) | 3.716 | 1.308 (~35 %) | Dataset maior; as máscaras de lesão são utilizadas para obter a localização da ROI quando disponíveis. |

- **Rótulo binário:** `Normal = 0` e `Anormal = 1`, agrupando casos benignos e malignos.
- Os números correspondem às imagens efetivamente carregadas do arquivo `.npz` utilizado pelo notebook, e não necessariamente à quantidade total de arquivos disponíveis nas bases originais.
- As imagens e seus rótulos são carregados junto com os identificadores de paciente, utilizados para evitar que imagens de uma mesma pessoa sejam distribuídas entre treino, validação e teste.

## 3. Arquitetura

```text
Mamografia original
      │
      ▼
[Pré-processamento] segmentação da mama → orientação padronizada
                   → recorte ROI → normalização → 224×224
      │
      ▼
[ResNet50 pré-treinada no ImageNet-1k]
      │
      ▼
[Camada classificadora] 1 neurônio de saída (probabilidade de "Anormal")
      │
      ▼
[Fine-tuning em 2 fases]
      1) treinamento apenas da cabeça
      2) descongelamento da rede e ajuste fino
      │
      ▼
[Escolha do limiar] feita na validação interna, sem utilizar o teste
      │
      ├── Exp. 1: validação cruzada 5-fold agrupada por paciente
      └── Exp. 2: treino em A e teste em B (e vice-versa), 3 sementes
      │
      ▼
[Análise] AUC, métricas de classificação, IC 95 % por bootstrap de pacientes,
          curvas ROC, matrizes de confusão e deslocamento de limiar
```

## 4. Metodologia 

### 4.1 Pré-processamento

O notebook utiliza um arquivo `.npz` com as imagens já pré-processadas, contendo imagens, rótulos e identificadores de paciente. O pipeline de preparação considera os seguintes passos:

1. **Segmentação da mama:** limiarização de Otsu, abertura morfológica, identificação do maior componente conexo e preenchimento de buracos para separar a mama do fundo.
2. **Padronização da orientação:** a parede torácica é posicionada à esquerda. A imagem é espelhada quando necessário; essa operação é uma padronização da orientação, não uma augmentation.
3. **Recorte da ROI:** no modo utilizado (`ROI_MODE = "roi"`), é extraída uma janela quadrada com lado equivalente a **30 % da altura da mama**.
   - **Com coordenada de lesão:** a janela é centralizada na lesão. No MIAS, a coordenada vem dos metadados; no DDSM, é obtida pelo centroide da maior região identificada na máscara.
   - **Sem coordenada:** o centro da janela é sorteado entre posições válidas dentro da mama, com semente fixa.
4. **Normalização e redimensionamento:** normalização min-max por imagem, padding para manter formato quadrado e redimensionamento para **224×224 pixels**.
5. **CLAHE:** o arquivo de dados utilizado é identificado como pré-processado com CLAHE. Essa etapa não é executada novamente durante o treinamento apresentado neste notebook.

### 4.2 Modelo

- **Arquitetura:** `resnet50`, da biblioteca `torchvision`, com pesos pré-treinados `IMAGENET1K_V1`.
- **Parâmetros:** aproximadamente **23,5 milhões**.
- **Cabeça classificadora:** a camada `fc` original é substituída por uma camada linear com **1 logit**, adequada à classificação binária.
- **Função de perda:** `BCEWithLogitsLoss`.
- **Entrada:** as imagens em escala de cinza são replicadas em três canais para atender ao formato esperado pelos pesos pré-treinados no ImageNet.

### 4.3 Treinamento (fine-tuning em duas fases)

O treinamento utiliza transferência de aprendizado em duas fases:

| Fase | Épocas | O que treina | Taxa de aprendizado |
|---|---:|---|---:|
| 1. Cabeça classificadora | 5 | Apenas a camada `fc`; backbone congelado | 3e-4 |
| 2. Fine-tuning | Até 25 adicionais, respeitando o limite total de 30 | Rede inteira, com taxas diferenciadas | Backbone: 1e-5; cabeça: 3e-4 |

Outros detalhes:

- **Otimizador:** AdamW, com `weight_decay = 0,05`.
- **Agendamento da taxa de aprendizado:** `CosineAnnealingLR`.
- **Desbalanceamento de classes:** `pos_weight` calculado somente com os dados de treinamento.
- **Estabilidade:** mixed precision (AMP) quando há GPU, gradient clipping em 1,0 e batch de 32 imagens.
- **Augmentation (somente no treino):** rotação de até ±10°, escala entre 0,9 e 1,1, translação de até ±5 % e variação de brilho/contraste de até ±10 %. **Não há flip horizontal.**
- **Early stopping:** monitora o AUC da validação interna, com paciência de 7 épocas e mínimo de 5 épocas. Ao final, restaura os melhores pesos.

### 4.4 Cuidados contra vazamento de dados

A divisão dos dados é feita por paciente, e não por imagem. Isso é importante porque imagens pertencentes à mesma pessoa podem apresentar características correlacionadas.

- **Divisão por paciente:** utiliza `StratifiedGroupKFold`, mantendo as imagens de um mesmo paciente no mesmo grupo. No MIAS, as duas mamas de uma paciente permanecem juntas.
- **Validação interna:** em cada fold, 15 % dos pacientes do conjunto de treino são separados para early stopping e escolha do limiar.
- **Teste isolado:** o conjunto de teste não participa da escolha da época, do limiar ou dos parâmetros do modelo.
- **Limiar de decisão:** é selecionado pelo método de Youden (ou F1, conforme configuração), utilizando somente as predições da validação interna.

### 4.5 Os dois experimentos

**Experimento 1: validação cruzada interna.** Para cada dataset, são realizados cinco folds com divisão agrupada por paciente. Em cada repetição, uma parte dos dados é reservada para teste, enquanto outra parte do conjunto de treinamento é separada para validação interna.

**Experimento 2: domain shift direto.** O modelo é treinado em um dataset e avaliado no dataset inteiro de destino, nos dois sentidos: MIAS → DDSM e DDSM → MIAS. O experimento é repetido com três sementes (`42`, `123` e `999`). O limiar é definido somente com a validação do dataset de origem.

### 4.6 Análise estatística

- **Drop de AUC:** diferença entre o AUC interno e o AUC obtido na avaliação cruzada entre datasets.
- **IC 95 %:** estimado por bootstrap com 1.000 reamostragens de pacientes, e não de imagens.
- **Predição cruzada agregada (ensemble):** média das probabilidades produzidas pelas três sementes.
- **Limiar oráculo:** limiar calculado com os rótulos do dataset de destino, utilizado apenas para análise diagnóstica do deslocamento do limiar. Não representa desempenho válido de generalização e não deve ser reportado como resultado do modelo.

## 5. Como executar

1. **Preparar os dados:** gerar o arquivo `dados_preprocessados_roi.npz` com o pipeline de pré-processamento utilizado no projeto. O arquivo deve conter `mias_X`, `mias_y`, `mias_g`, `ddsm_X`, `ddsm_y` e `ddsm_g`.
2. **Enviar ao Google Drive:** colocar o `.npz` no caminho configurado em `NPZ_PATH` e ajustar os caminhos dos datasets e diretórios de saída em `CONFIG`, se necessário.
3. **Abrir no Google Colab:** executar o notebook com GPU habilitada e rodar as células em ordem.
4. **Teste rápido:** definir `QUICK_TEST = True` para executar uma versão reduzida (2 épocas, 1 semente e 100 reamostragens de bootstrap) antes da execução completa.

**Dependências:** `torch`, `torchvision`, `albumentations`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib` e `Pillow`.


## 6. Glossário 

- **ResNet50:** arquitetura de rede neural convolucional residual com 50 camadas.
- **CNN:** rede neural convolucional, arquitetura que utiliza filtros convolucionais para aprender características das imagens.
- **Fine-tuning:** ajuste de um modelo pré-treinado para uma nova tarefa.
- **AUC-ROC:** capacidade de separar as classes independentemente de um limiar específico. 0,5 equivale ao acaso e 1,0 representa separação perfeita.
- **Sensibilidade (recall):** entre os casos realmente anormais, quantos o modelo identificou.
- **Especificidade:** entre os casos realmente normais, quantos o modelo reconheceu como normais.
- **Precisão:** entre os casos classificados como anormais, quantos eram realmente anormais.
- **F1-score:** média harmônica entre precisão e sensibilidade.
- **Limiar:** valor de corte da probabilidade utilizado para classificar uma imagem como Normal ou Anormal.
- **Domain shift:** diferença na distribuição dos dados entre os datasets de treinamento e avaliação.
- **CLAHE:** técnica de equalização adaptativa de contraste local.
- **ROI:** região de interesse; neste notebook, uma janela quadrada recortada da mama.
- **Bootstrap:** técnica de reamostragem utilizada para estimar intervalos de confiança.
