# Dataset de mamografias (mini-MIAS + mini-DDSM)

## Resumo

- **Script:** transforma as mamografias do mini-MIAS e do mini-DDSM em um dataset para classificar **normal (0) × anormal (1)**, reduzindo atalhos que inflam o resultado (*shortcut learning* e *data leakage*).
- **Como:** `python gerar_dataset.py` (pode rodar em CPU, sem GPU).
- **Resultado:** um único arquivo compactado, `dados_preprocessados_full.npz`, com os dois datasets juntos, separados pelo prefixo `mias_` e `ddsm_`.

## Como usar

```bash
pip install numpy pandas pillow scipy matplotlib joblib scikit-image
python gerar_dataset.py             # gera o dataset
python gerar_dataset.py --amostras  # (opcional) figuras para conferir o resultado
```

Antes de rodar, ajuste os caminhos no `CONFIG` do script: `MIAS_DIR`, `MIAS_INFO` e `DDSM_DIR`.

## Dados de entrada

| Dataset | Rótulo vem de | Coordenada da lesão vem de |
|---|---|---|
| mini-MIAS | `mias_info.csv` (classe NORM = normal) | colunas `x`, `y` do arquivo |
| mini-DDSM | nome da pasta (`Normal`, `Benign`, `Cancer`) | centro da máscara da lesão (`*_Mask.jpg`) |

Benigno e câncer viram a mesma classe: **anormal (1)**.

## O que o pipeline faz

1. **Metadados:** monta uma tabela com caminho, rótulo, paciente e coordenada da lesão.
2. **Filtros:** mantém só os sites `A` e `D` do DDSM e descarta anormais sem lesão localizada.
3. **Segmentação:** separa a mama do fundo (Otsu) e remove o músculo peitoral.
4. **Recorte:** mama inteira (`full`) ou janela quadrada (`roi`), conforme `ROI_MODE`.
5. **Alinhamento:** espelha para a mama ficar sempre do mesmo lado.
6. **Normalização:** CLAHE (equalização adaptativa de contraste).
7. **Padronização:** completa para quadrado e redimensiona para 224 x 224.
8. **Cache:** só reprocessa se algum parâmetro mudar.

### `full` × `roi`

| | `full` | `roi` |
|---|---|---|
| Anormais | mama inteira | janela **centrada na lesão** |
| Normais | mama inteira | janela em posição **pareada** com lesões reais |
| Detalhe da lesão | menor (mama toda reduzida a 224 px) | maior (só a janela é reduzida) |
| Formato do resultado | `(N, 224, 224)` | `(N, 224, 224)` |

## Mitigação de atalhos

| Atalho possível | Medida |
|---|---|
| Brilho/contraste diferente entre exames | CLAHE |
| Equipamentos diferentes (sites do DDSM) | só sites A e D |
| Janela em posição diferente entre normais e anormais | pareamento de posição anatômica |
| Peitoral e lado da mama | remoção do peitoral e espelhamento |
| Vazamento entre treino e teste | saída com ID de paciente (`*_g`) |

## Resultado

Um **único arquivo compactado** com os dois datasets, separados pelo prefixo:

| Array | Conteúdo |
|---|---|
| `mias_X`, `ddsm_X` | imagens `uint8`, formato `(N, 224, 224)` |
| `mias_y`, `ddsm_y` | rótulos (0 = normal, 1 = anormal) |
| `mias_g`, `ddsm_g` | paciente de cada imagem |

O programa que usar o arquivo sabe se a imagem é MIAS ou DDSM **pelo prefixo** (`mias_` ou `ddsm_`). Os datasets não são misturados: cada um tem seu próprio tamanho, e a posição `i` de `X`, `y` e `g` é sempre a mesma imagem.

```python
import numpy as np
d = np.load("dados_preprocessados_full.npz")
X_mias, y_mias, g_mias = d["mias_X"], d["mias_y"], d["mias_g"]
X_ddsm, y_ddsm, g_ddsm = d["ddsm_X"], d["ddsm_y"], d["ddsm_g"]
```

Para usar os dois juntos, concatene os arrays e crie você mesmo uma coluna de origem, pois o arquivo não traz uma:

```python
X = np.concatenate([d["mias_X"], d["ddsm_X"]])
y = np.concatenate([d["mias_y"], d["ddsm_y"]])
g = np.concatenate([d["mias_g"], d["ddsm_g"]])
origem = np.array(["mias"] * len(d["mias_y"]) + ["ddsm"] * len(d["ddsm_y"]))
```

### Outros arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `roi_report.csv` | uma linha por imagem (caminho, rótulo, paciente, `roi_x`, `roi_y`, `dataset`, `has_roi`), na mesma ordem dos arrays |
| `cache/*.npy` | imagens pré-processadas, para não reprocessar |
| `amostras_mias.png`, `amostras_ddsm.png` | figuras de conferência (`--amostras`): caixa vermelha = mama, verde = lesão, amarela = janela pareada de um normal |

## Validação do dataset (`validar_img.py`)

Script de **auditoria**: não gera o dataset, confere se o que o gerador produziu está correto e sem atalhos. Deve ficar na **mesma pasta** do `gerar_dataset.py`, pois reutiliza o `CONFIG` e as funções dele (assim vê exatamente o que foi para o `.npz`).

```bash
python validar_img.py              # roda tudo
python validar_img.py --auditoria  # só os números
python validar_img.py --figuras    # só as figuras sobre a imagem original
python validar_img.py --npz        # só as checagens do .npz
```

| Etapa | O que verifica |
|---|---|
| Auditoria | anormais sem coordenada de lesão; lesão fora ou deslocada da janela; máscara da mama suspeita; janelas com muito fundo preto (normais x anormais); origem do DDSM (letra do caso) x classe |
| Figuras | imagem original com a mama (vermelho), a janela (verde = com lesão, amarelo = sem lesão) e a lesão (ciano/magenta), ao lado do que vai para o `.npz` |
| Checagens do `.npz` | tamanhos consistentes, imagens duplicadas, médias por classe e grade com 8 imagens por classe |
| Testes de atalho | AUC com validação **por paciente** usando só a posição da janela e só estatísticas de brilho, mais um baseline trivial (imagem 14x14 + regressão logística) |

**Como interpretar a AUC:** perto de 0,50 é o esperado (sem atalho); acima de 0,65 com a posição da janela indica que o recorte entrega o rótulo; se o baseline trivial já chega perto do seu modelo, o problema está nos dados, não na arquitetura. Os testes de atalho exigem `scikit-learn` (opcional).

Saídas: `diagnostico_mias.csv`, `diagnostico_ddsm.csv` (uma linha por imagem, com métricas e alertas), `fig_<dataset>_<categoria>.png` e `npz_<dataset>.png`.

## Principais parâmetros (`CONFIG`)

| Parâmetro | Função |
|---|---|
| `ROI_MODE` | `full` ou `roi` |
| `ROI_SIZE_FRAC` | tamanho da janela ROI (fração da altura da mama) |
| `NORM_MODE` | `clahe` (padrão), `pct` ou min-max |
| `DDSM_SITES` | sites do DDSM mantidos |
| `DROP_ABNORMAL_NO_ROI` | descarta anormais sem lesão localizada |
| `MATCH_SPATIAL_DIST` | liga o pareamento anatômico dos normais |
| `REMOVE_PECTORAL` | liga a remoção do peitoral |
| `OUT_NPZ` | nome do arquivo final |
| `ROI_SEED` | semente para resultados reprodutíveis |

## Observações para uso deste script:

- **Divida sempre por paciente** (use `*_g`, ex.: `GroupKFold`), nunca por imagem. As duas mamas da mesma pessoa são muito parecidas e causam vazamento.
- **Trocou de `full` para `roi`?** Mude `OUT_NPZ`, senão o arquivo anterior é sobrescrito.
- Os IDs em `*_g` indicam a origem (`mias_001`; no DDSM, `Normal_A_0001_1`, `Benign_...`, `Cancer_...`).
