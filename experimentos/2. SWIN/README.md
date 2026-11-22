# Swin-T para classificação de mamografias

## Resumo 

Esses notebooks treinam um **Swin Transformer Tiny** (pré-treinado no ImageNet) para classificar se uma mamografia é **Normal** ou **Anormal** (benigno ou maligno). O foco não é só acertar dentro de um banco de dados. O foco é medir quanto o desempenho cai quando o modelo é testado em outro banco, dessa forma validando se o modelo está conseguindo de fato generalizar ou apenas decorar. 

Para isso, o notebook usa dois datasets públicos (**mini-MIAS** e **Mini-DDSM**) e faz dois experimentos:

1. **Teste interno:** treina e testa no mesmo dataset.
2. **Teste externo:** treina em um dataset e testa no outro, nos dois sentidos.

---

## 1. Generalização

Modelos de deep learning para mamografia costumam ter ótimos resultados no dataset em que foram treinados, contudo não conseguem manter tais resultados com outras imagens. Esse fenômeno é chamado de **domain shift** (mudança de domínio). Um modelo que só funciona no banco em que foi treinado tem pouca utilidade clínica.

Este notebook mede esse efeito de forma controlada e sem vazamento de dados, e responde a três perguntas:

- Quanto o AUC cai ao trocar de dataset?
- Essa queda é estatisticamente relevante ou pode ser acaso?
- O limiar de decisão escolhido na origem continua adequado no destino?

## 2. Dados

| Dataset | Imagens | Anormais | Observação |
|---|---|---|---|
| **mini-MIAS** | 318 | 111 (~35 %) | Dataset pequeno, digitalização antiga. Duas mamas por paciente. |
| **Mini-DDSM** (Complete, JPEG) | 3.716 | 1.308 (~35 %) | Dataset maior. Possui máscaras de lesão (`_Mask.jpg`). |

- **Rótulo binário:** `Normal = 0`. `Anormal = 1`, que agrupa benigno e maligno.
- Os números acima são os efetivamente carregados do arquivo `.npz` de dados pré-processados (saída do notebook). O texto introdutório do notebook cita 322 e ~7.808 imagens. Essa diferença vem das imagens descartadas na geração do `.npz`, como máscaras e overlays, ou sem metadados.

## 3. Arquitetura

```
Mamografia original
      │
      ▼
[Pré-processamento]  segmenta a mama → recorta uma janela (ROI) 224×224 → CLAHE
      │
      ▼
[Swin-T pré-treinado]  1 neurônio de saída (probabilidade de "Anormal")
      │
      ▼
[Fine-tuning em 2 fases]  1) só a cabeça   2) rede inteira com LR diferenciada
      │
      ▼
[Escolha do limiar]  feita numa validação interna, nunca no teste
      │
      ├── Exp. 1: validação cruzada 5-fold dentro de cada dataset
      └── Exp. 2: treina em A, testa em B (e vice-versa), 3 sementes
      │
      ▼
[Análise]  AUC, IC 95 % por bootstrap de pacientes, ROC, matriz de confusão, deslocamento de limiar
```

## 4. Metodologia em detalhe

### 4.1 Pré-processamento (igual nos dois datasets)

O pré-processamento é feito uma única vez pelo script externo `gerar_dataset_roi.py`. As funções aparecem no notebook só como documentação. O resultado fica em um arquivo `.npz`, que o notebook carrega.

1. **Segmentação da mama:** limiarização de Otsu, abertura morfológica, maior componente conexo e preenchimento de buracos. Isso separa a mama do fundo preto.
2. **Padronização da orientação:** a parede torácica é colocada sempre à esquerda. A imagem é espelhada se necessário, e esse espelhamento é uma padronização, não uma augmentation.
3. **Recorte de uma janela quadrada (ROI):** o lado da janela é **30 % da altura da mama**, e a janela sempre fica dentro do tecido mamário.
   - **Com coordenada de lesão:** a janela é centrada na lesão.
     - *MIAS:* o centro da anomalia vem do arquivo de metadados.
     - *DDSM:* é o centroide da maior lesão na máscara.
   - **Sem coordenada** (normais e anormais do DDSM sem máscara): a janela tem o mesmo tamanho, com centro **sorteado dentro da mama**, usando semente fixa.
4. **Normalização e redimensionamento:** min-max por imagem e resize para **224×224**.
5. **CLAHE** (equalização adaptativa de contraste): é aplicado dentro do script gerador. O código do CLAHE e seus parâmetros não estão no notebook, só no script.

### 4.2 Modelo

- **Arquitetura:** `swin_tiny_patch4_window7_224` (biblioteca `timm`), com **27,5 milhões de parâmetros** e pesos pré-treinados no ImageNet.
- **Cabeça:** trocada por **1 logit**, treinada com `BCEWithLogitsLoss`.
- A imagem em escala de cinza é replicada em 3 canais para casar com o formato esperado pelo modelo pré-treinado.

### 4.3 Treinamento (fine-tuning em duas fases)

| Fase | Épocas | O que treina | Taxa de aprendizado |
|---|---|---|---|
| 1. Cabeça | 5 | Só a camada de classificação | 3e-4 |
| 2. Fine-tuning | até 25 | Rede inteira | Backbone 1e-5, cabeça 3e-4 |

Outros detalhes:

- **Otimizador:** AdamW (weight decay 0,05) com cosine annealing.
- **Desbalanceamento de classes:** `pos_weight` calculado somente no conjunto de treino.
- **Estabilidade:** mixed precision (AMP), gradient clipping em 1,0 e batch de 32.
- **Augmentation (só no treino):** pequenas rotações (±10°), escala (0,9–1,1), translação (±5 %) e brilho/contraste (±10 %). **Não há flip horizontal.**
- **Early stopping:** monitora o AUC da validação interna, com paciência de 7 épocas. Ao final, restaura os melhores pesos.

### 4.4 Cuidados contra vazamento de dados

Este é o ponto mais importante da metodologia:

- **Divisão por paciente:** nenhum paciente aparece em dois conjuntos ao mesmo tempo (`StratifiedGroupKFold`, verificado com `assert`). No MIAS, as duas mamas de uma mesma paciente ficam juntas.
- **Validação interna separada:** em cada treino, 15 % dos pacientes viram uma validação usada para early stopping e para escolher o limiar de decisão.
- **O teste nunca influencia nada:** nem época, nem limiar, nem hiperparâmetro. No teste externo, nada do dataset de destino é usado no treino.
- **Limiar (Youden):** é o ponto da curva ROC que maximiza sensibilidade + especificidade − 1, escolhido **só na validação da origem**.

### 4.5 Os dois experimentos

**Experimento 1: validação cruzada interna.** Para cada dataset, usa 5-fold agrupada por paciente. Mostra o desempenho "ideal", sem mudança de domínio.

**Experimento 2: domain shift direto.** Treina em um dataset e avalia o **dataset inteiro** do outro, nos dois sentidos (MIAS → DDSM e DDSM → MIAS). Repete com 3 sementes (42, 123, 999).

### 4.6 Análise estatística

- **Drop** = AUC interno − AUC externo.
- **IC 95 % por bootstrap de pacientes** (1.000 reamostragens). Reamostra *pacientes* e não imagens, porque imagens do mesmo paciente são correlacionadas.
- **Ensemble:** para o resultado externo agregado, tira a média das probabilidades das 3 sementes.
- **Deslocamento de limiar:** compara sensibilidade e especificidade no destino com o limiar da origem e com um limiar "oráculo", que usa os rótulos do destino. O oráculo serve **só para diagnóstico** e não deve ser reportado como desempenho.


## 5. Como executar

1. **Gerar os dados:** rode o script `gerar_dataset_roi.py` (fora deste notebook) nos datasets originais. Ele produz `dados_preprocessados_roi.npz`, com `mias_X/y/g` e `ddsm_X/y/g`.
2. **Enviar ao Google Drive:** suba o `.npz` e ajuste `NPZ_PATH` e `OUTPUT_DIR` em `CONFIG`.
3. **Abrir no Google Colab** com GPU e executar as células em ordem.
4. **Teste rápido:** use `QUICK_TEST = True` (2 épocas, 1 semente, 100 bootstraps) para verificar se tudo roda antes da execução completa. O Mini-DDSM é demorado.

**Dependências:** `torch`, `timm`, `albumentations`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `Pillow`.


## 6. Glossário

- **AUC-ROC:** capacidade de separar as classes, independentemente do limiar. 0,5 é chute e 1,0 é perfeito.
- **Sensibilidade (recall):** entre os casos realmente anormais, quantos o modelo encontrou.
- **Especificidade:** entre os casos realmente normais, quantos o modelo reconheceu como normais.
- **Limiar:** valor de corte da probabilidade que separa "Normal" de "Anormal".
- **Domain shift:** diferença de distribuição entre os dados de treino e os de uso real.
- **CLAHE:** técnica de realce de contraste local em imagens.
- **ROI:** região de interesse, aqui uma janela quadrada recortada da mama.
- **Bootstrap:** técnica de reamostragem para estimar intervalos de confiança.
