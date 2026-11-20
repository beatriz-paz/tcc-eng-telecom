# Dual-Branch (ResNet50 ‖ Swin-T) para classificação de mamografias

Classificação binária (normal × anormal) de mamografias com um modelo de dois ramos em paralelo (CNN + Transformer), avaliando a **generalização entre datasets** (mini-MIAS ↔ Mini-DDSM).

---

## 1. Breve resumo

Os 5 notebooks usam **os mesmos dados, o mesmo protocolo e o mesmo pipeline de treino**. O que muda é só o modelo: a forma de reduzir resolução (pooling) e a forma de adaptar os pesos pré-treinados.

| # | Notebook | Pooling (redução de resolução) | Adaptação dos pesos |
|---|---|---|---|
| 01 | `01_dual_branch_roi_clahe_finetuning` | Original (`maxpool` + `PatchMerging`) | Fine-tuning em 2 fases |
| 02 | `02_dual_branch_roi_wavelet_clahe_finetuning` | Wavelet **fixa** (db4 + Haar) | Fine-tuning em 2 fases |
| 03 | `03_dual_branch_roi_clahe_ssf` | Original | **SSF** (backbones congelados) |
| 04 | `04_dual_branch_roi_wavelet_clahe_ssf` | Wavelet **fixa** (db4 + Haar) | **SSF** |
| 05 | `05_dual_branch_roi_wavelet_clahe_finetuning_trainable` | Wavelet **treinável** (db4 + Haar) | Fine-tuning em 2 fases |

Estrutura de cada notebook: Setup → Config → Dados → Modelo → Treino → Exp. 1 (CV interna) → Exp. 2 (cross-dataset) → Análise.

---

## 2. Dados

| Dataset | Imagens | Normais | Anormais |
|---|---|---|---|
| mini-MIAS | 322 | 208 | 114 |
| Mini-DDSM | 3.716 | 2.408 | 1.308 |

**Rótulos:** Normal = 0; Benigno e Câncer = 1 (DDSM). No MIAS, qualquer classe diferente de `NORM` = 1.

**Pré-processamento** (igual nos dois datasets, gerado por `gerar_dataset_roi.py` e carregado de um `.npz`):
1. Segmentação da mama (Otsu) e recorte.
2. Parede torácica sempre à esquerda (espelha se necessário).
3. Normalização min-max, padding para quadrado e resize para **224×224**.
4. **ROI:** janela quadrada de 30 % da altura da mama.
   - Com coordenada da lesão: janela centrada nela (MIAS: `mias_info`; DDSM: centroide da máscara).
   - Sem coordenada: janela do mesmo tamanho em posição sorteada no tecido (semente fixa).
5. **CLAHE** aplicado na geração do dataset.

**Augmentation (treino):** affine leve (escala 0,9–1,1; translação ±5 %; rotação ±10°) e brilho/contraste (±10 %). Normalização ImageNet.

---


## 3. Modelos

### 01: Baseline (fine-tuning)
Dual-branch original, sem modificações na arquitetura. Serve de referência.

### 02 e 05: Subamostragem por wavelet
A redução de resolução é trocada por uma DWT 2D de 1 nível:

| Ramo | Original | Substituído por |
|---|---|---|
| ResNet50 | `maxpool` do stem (112→56) | `WaveletPool2d4Band` com **db4**: 4 sub-bandas (LL, LH, HL, HH) + conv 1×1 (4C→C), iniciada copiando só o LL |
| Swin-T | concat 2×2 do `PatchMerging` (3 camadas) | **Haar 2D** (LL, LH, HL, HH) seguida de `norm` e `reduction` pré-treinados. Sem parâmetros novos |

- **02:** filtros wavelet **fixos** (buffers, não treinam).
- **05:** filtros **treináveis** (`nn.Parameter` iniciados em db4/Haar): 16 parâmetros no ramo CNN e 12 no Swin (3 camadas × 4).

### 03 e 04: SSF (Scaling and Shifting your Features)
Backbones **congelados**. Cada camada recebe um módulo `y = γ·x + β` (por canal), com `γ=1` e `β=0` no início (o modelo começa idêntico ao pré-treinado).

- **ResNet50:** SSF após cada `BatchNorm2d`.
- **Swin-T:** SSF após cada `Linear`, `LayerNorm` e a conv do patch embedding.
- **Treináveis:** `γ` e `β` dos dois ramos + `head` (+ conv 1×1 da wavelet no 04).
- `fuse_ssf()` (opcional) incorpora `γ`/`β` nas camadas após o treino, sem custo extra na inferência.
- Cada notebook tem `ADAPT_MODE` (`"ssf"` ou `"finetune"`) para alternar entre os métodos.

---

## 4. Treinamento

| Item | Configuração |
|---|---|
| Otimizador | AdamW + cosine annealing |
| Loss | `BCEWithLogitsLoss` com `pos_weight` (calculado só no treino) |
| Épocas / batch | 30 / 32 |
| Weight decay | 0,05 |
| Precisão / estabilidade | AMP (GPU) e gradient clipping = 1,0 |
| Early stopping | Em `val_auc`, paciência 7, mínimo 5 épocas, restaura os melhores pesos |
| Limiar de decisão | Youden (ou F1), escolhido **só na validação interna** |
| Sementes | 42, 123, 999 |

---

## 5. Protocolo de avaliação

1. **Exp. 1, validação cruzada interna:** 5-fold com `StratifiedGroupKFold`, agrupado por paciente (nenhum paciente em dois conjuntos, checado por `assert`). Em cada fold, o treino é dividido por paciente em treino e validação interna (15 %).
2. **Exp. 2, domain shift:** treina em um dataset e testa no outro inteiro, nos dois sentidos (MIAS→DDSM e DDSM→MIAS), com 3 sementes. O limiar vem só da origem.

**Métricas:** AUC-ROC (principal), sensibilidade, especificidade, acurácia balanceada, F1, precisão e acurácia.

---

## 6. Como executar

1. Gere o dataset com `gerar_dataset_roi.py` e coloque o `.npz` no Google Drive (`NPZ_PATH`).
2. Ajuste os caminhos em `CONFIG` e rode os notebooks no Colab (GPU recomendada).
3. Use `QUICK_TEST = True` para um teste rápido (2 épocas, 1 semente, 100 bootstraps) antes da execução completa.
4. As saídas (CSVs, gráficos e `.pkl`) são salvas em pastas separadas por variante, sem sobrescrever as outras.

**Dependências:** `torch`, `torchvision`, `timm`, `albumentations`, `scikit-learn`, `scipy`, `joblib`, `pandas`, `matplotlib`, `seaborn`.
