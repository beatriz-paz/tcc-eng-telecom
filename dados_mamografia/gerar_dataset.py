"""Gera o dataset pré-processado com ROI (mini-MIAS + mini-DDSM) sem atalhos/data leakage.

Utiliza Pareamento Empírico 1:1 com Projeção Anatômica Restrita.
Roda no computador, onde estão as imagens. Sem GPU.
Dependências: pip install numpy pandas pillow scipy matplotlib joblib scikit-image
"""

# ===== VISÃO GERAL DO PIPELINE =====
# 1) Lê metadados (caminho, rótulo normal/anormal, paciente, coordenada da lesão) de MIAS e DDSM
# 2) Filtra para reduzir atalhos (sites de DDSM, anormais sem ROI)
# 3) Segmenta a mama, recorta (mama inteira ou janela ROI), alinha o lado, normaliza (CLAHE) e redimensiona
# 4) Salva imagens (X), rótulos (y) e grupos de paciente (g) em .npz + relatório CSV

import sys, re, json, time, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from joblib import Parallel, delayed

# Todas as configurações do pipeline ficam neste dicionário
CONFIG = dict(
    # caminhos
    # Pastas das imagens (mini-MIAS e mini-DDSM) e arquivo de informações do MIAS
    MIAS_DIR   = r"C:\Users\Beatriz\Documents\TCC\mini-MIAS\MIAS",
    MIAS_INFO  = r"C:\Users\Beatriz\Documents\TCC\mini-MIAS\mias_info.csv",
    DDSM_DIR   = r"C:\Users\Beatriz\Documents\TCC\mini-DDSM\MINI-DDSM-Complete-JPEG-8",
    # DDSM: nome da pasta -> rótulo binário (0 = normal, 1 = anormal)
    DDSM_CLASS_DIRS  = {"Normal": 0, "Benign": 1, "Cancer": 1},   # pasta -> rótulo
    # Arquivos com esses termos no nome são máscaras/overlays, não mamografias
    EXCLUDE_PATTERNS = ["mask", "overlay"],
    # Pasta de cache dos arrays pré-processados (evita reprocessar)
    CACHE_DIR  = "./cache",
    # Arquivo final com o dataset
    OUT_NPZ    = "dados_preprocessados_full.npz",

    # pré-processamento livre de atalhos
    # Lado (px) da imagem final quadrada
    IMG_SIZE         = 224,
    # 'full' = recorta a mama inteira; 'roi' = recorta janela quadrada (lesão nos anormais, posição pareada nos normais)
    ROI_MODE         = "full",     # 'full' (mama inteira) ou 'roi' (janela quadrada)
    # Lado da janela ROI como fração da altura da mama
    ROI_SIZE_FRAC    = 0.30,      # lado da janela = fração da altura da mama
    # Semente base para sorteios reprodutíveis
    ROI_SEED         = 2024,
    # Maior lado da máscara reduzida usada para segmentar a mama (acelera o processamento)
    MASK_MAX_SIDE    = 256,
    # Limiar de fundo = fator x limiar de Otsu
    BG_THRESH_FACTOR = 0.5,
    # Completa com zeros até ficar quadrada antes de redimensionar (evita distorcer a imagem)
    PAD_TO_SQUARE    = True,
    
    # --- CORREÇÕES DE ATALHO (SHORTCUT MITIGATION) ---
    # CLAHE (equalização adaptativa de contraste) uniformiza o brilho entre exames
    NORM_MODE        = "clahe",   # 'clahe' (elimina atalho de brilho)
    # Percentis de corte (usados apenas se NORM_MODE = 'pct')
    NORM_PCT         = (1, 99),
    # Mantém só os sites A e D do DDSM
    DDSM_SITES       = ["A", "D"],# Elimina sites B e C para evitar viés de equipamento
    # Remove anormais sem coordenada de lesão (não há onde centrar a ROI)
    DROP_ABNORMAL_NO_ROI = True,  # descarta anormais sem máscara
    # Normais recebem janela em posição anatômica semelhante à de lesões reais (evita viés de posição)
    MATCH_SPATIAL_DIST   = True,  # Pareamento Empírico 1:1 com Projeção Anatômica Restrita
    # Tenta remover o músculo peitoral (região clara no topo da mama)
    REMOVE_PECTORAL      = True,  # Máscara/descarta músculo peitoral em MLO
    # -1 = usa todos os núcleos da CPU
    N_JOBS = -1,
)
# Cria a pasta de cache, se não existir
Path(CONFIG["CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

# Extensões de imagem aceitas
IMG_EXTS = {".png", ".jpg", ".jpeg", ".pgm", ".tif", ".tiff", ".bmp"}


# Lista recursivamente todas as imagens de uma pasta
def index_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS]


# True se o nome do arquivo indica máscara/overlay
def is_excluded(path: Path):
    name = path.name.lower()
    return any(pat in name for pat in CONFIG["EXCLUDE_PATTERNS"])


# Monta os metadados do mini-MIAS (rótulo, paciente e coordenada da lesão por imagem)
def load_mias_metadata():
    # Mapa nome-do-arquivo -> caminho
    files = {p.stem.lower(): p for p in index_images(CONFIG["MIAS_DIR"])}
    # Acumula, por imagem, o rótulo e a coordenada da lesão
    rows = {}

    def num(v):
        try: return float(v)
        except Exception: return np.nan

    # Registra uma linha de anotação; imagem com qualquer anomalia é anormal; guarda só a 1ª coordenada de lesão
    def add(ref, cls, x, y):
        abnormal = int(str(cls).strip().upper() not in {"NORM", "NORMAL", "0"})
        r = rows.setdefault(ref, dict(label=0, roi_x=np.nan, roi_y=np.nan))
        r["label"] = max(r["label"], abnormal)
        if abnormal and np.isnan(r["roi_x"]):
            r["roi_x"], r["roi_y"] = x, y

    # Lê o arquivo de informações (.csv ou .txt do MIAS original)
    info = Path(CONFIG["MIAS_INFO"])
    if info.suffix.lower() == ".csv":
        tab = pd.read_csv(info)
        tab.columns = [str(c).strip().lower() for c in tab.columns]
        # Descobre automaticamente a coluna de referência da imagem (mdbXXX)
        ref_col = next((c for c in ["refnum", "ref", "reference", "image", "filename", "file", "id", "name"]
                        if c in tab.columns), tab.columns[0])
        cls_col = next((c for c in ["class", "abnormality", "label"] if c in tab.columns), None)
        for _, r in tab.iterrows():
            ref = Path(str(r[ref_col]).strip()).stem.lower()
            if ref.startswith("mdb"):
                add(ref, r[cls_col], num(r.get("x")), num(r.get("y")))
    else:
        # Formato .txt original: ref, tecido, classe, severidade, x, y, raio
        with open(info, "r", errors="ignore") as f:
            for line in f:
                t = line.split()
                if not t or not t[0].lower().startswith("mdb"): continue
                cls = t[2] if len(t) > 2 else "NORM"
                x, y = (num(t[4]), num(t[5])) if len(t) >= 7 else (np.nan, np.nan)
                add(t[0].lower(), cls, x, y)

    # Junta anotações e arquivos de imagem existentes em uma tabela
    recs = []
    for ref, r in sorted(rows.items()):
        p = files.get(ref)
        if p is None: continue
        w, h = Image.open(p).size
        # MIAS tem origem no canto inferior esquerdo; converte y para origem no topo
        roi_y = (h - r["roi_y"]) if not np.isnan(r["roi_y"]) else np.nan
        recs.append(dict(path=str(p), label=r["label"],
                         # Paciente = par de imagens (esq./dir.) com mesmo grupo; usado depois para separar treino/teste sem vazamento
                         patient=f"mias_{(int(ref[3:]) + 1) // 2:03d}",
                         roi_x=r["roi_x"], roi_y=roi_y))
    return pd.DataFrame(recs)


# Monta os metadados do mini-DDSM a partir da estrutura de pastas (rótulo = pasta; sem coordenada ainda)
def load_ddsm_metadata():
    root = Path(CONFIG["DDSM_DIR"])
    rows = []
    # Percorre cada pasta de classe
    for cls_name, lab in CONFIG["DDSM_CLASS_DIRS"].items():
        cdir = root / cls_name
        if not cdir.exists(): continue
        for p in sorted(index_images(cdir)):
            # Ignora máscaras/overlays
            if is_excluded(p): continue
            rel = p.relative_to(cdir).parts
            # ID do paciente: nome da subpasta ou prefixo do nome do arquivo
            pid = rel[0] if len(rel) >= 2 else (re.match(r"^([A-Za-z]_\d+_\d+)", p.stem) or [None, p.stem])[1]
            rows.append(dict(path=str(p), label=lab, patient=f"{cls_name}_{pid}",
                             roi_x=np.nan, roi_y=np.nan))
    return pd.DataFrame(rows)


# Transforma a máscara da lesão (frequentemente só um contorno) em uma região sólida
def lesion_region(m):
    # f = fator de redução para máscaras muito grandes
    f = max(1, int(max(m.shape) // 700))
    h, w = (m.shape[0] // f) * f, (m.shape[1] // f) * f
    # Reduz a máscara em blocos fxf (bloco vale 1 se tiver algum pixel ligado)
    ms = m[:h, :w].reshape(h // f, f, w // f, f).any(axis=(1, 3))
    out = ms
    # Fecha lacunas com intensidade crescente até a região ficar compacta (solidez >= 0.3)
    for it in (2, 6, 14, 30):
        pad = it + 2
        # Fechamento morfológico (com margem para não cortar bordas)
        c = ndi.binary_closing(np.pad(ms, pad), structure=np.ones((3, 3)), iterations=it)[pad:-pad, pad:-pad]
        # Preenche buracos internos
        c = ndi.binary_fill_holes(c)
        # Rotula regiões conectadas
        lab, n = ndi.label(c)
        if n == 0: continue
        # Tamanho de cada região (ignora o fundo)
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        # Maior região e sua solidez = área / área da caixa envolvente
        ys, xs = np.nonzero(lab == sz.argmax())
        solidity = sz.max() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        out = c
        if solidity >= 0.3: break
    return out, f


# Para cada imagem anormal do DDSM, acha a máscara e calcula o ponto central da lesão (roi_x, roi_y)
def derive_ddsm_roi(df):
    df = df.copy()
    for i, r in df.iterrows():
        # Normais não têm lesão
        if r.label == 0: continue
        p = Path(r.path)
        # Nomes possíveis do arquivo de máscara
        cands = [p.with_name(f"{p.stem}_Mask.jpg"), p.with_name(f"Mask_{p.stem}.jpg")]
        mask_path = next((c for c in cands if c.exists()), None)
        if mask_path is None:
            # Plano B: procura qualquer arquivo 'mask' com o nome da imagem
            mask_path = next((f for f in p.parent.iterdir()
                              if "mask" in f.name.lower() and p.stem.lower() in f.name.lower()), None)
        if mask_path is None: continue

        # Lê a máscara em cinza; descarta se estiver vazia
        m = np.asarray(Image.open(mask_path).convert("L")).astype(np.float32)
        if m.max() < 128: continue
        # Binariza; se mais da metade for 'ligada', a máscara está invertida e é desinvertida
        m = m > 0.5 * m.max()
        if m.mean() > 0.5: m = ~m
        # Região sólida da lesão
        reg, f = lesion_region(m)
        # Mantém apenas o maior componente conectado
        lab, n = ndi.label(reg)
        if n == 0: continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        comp = lab == sizes.argmax()
        # Centro de massa da lesão e o pixel da lesão mais próximo dele (garante ponto dentro da lesão)
        cy_s, cx_s = ndi.center_of_mass(comp)
        pts = np.argwhere(comp)
        py, px = pts[np.argmin(((pts - [cy_s, cx_s]) ** 2).sum(1))]
        cy_m, cx_m = (py + 0.5) * f, (px + 0.5) * f

        # Converte para as coordenadas da imagem original
        W, H = Image.open(p).size
        df.at[i, "roi_x"] = cx_m * W / m.shape[1]
        df.at[i, "roi_y"] = cy_m * H / m.shape[0]
    return df


# Limiar de Otsu: separa fundo e tecido maximizando a variância entre classes
def otsu_threshold(x01, nbins=256):
    hist, edges = np.histogram(x01, bins=nbins, range=(0, 1))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    m = np.cumsum(p * centers)
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m) ** 2 / (w0 * w1)
    return centers[np.nanargmax(np.nan_to_num(between))]


# Segmenta a mama e retorna caixa (y0, y1, x0, x1), máscara reduzida e escala
def breast_geometry(img01, cfg):
    H, W = img01.shape
    # Reduz a imagem para segmentar mais rápido
    scale = min(1.0, cfg["MASK_MAX_SIDE"] / max(H, W))
    small = np.asarray(Image.fromarray(img01).resize((max(1, int(W * scale)), max(1, int(H * scale))),
                                                     Image.BILINEAR))
    # Tecido = pixels acima do limiar de fundo
    mask = small > cfg["BG_THRESH_FACTOR"] * otsu_threshold(small)
    # Abertura remove ruídos pequenos
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)), iterations=2)
    # Mantém só o maior componente (a mama) e preenche buracos
    lab, n = ndi.label(mask)
    # Falha na segmentação: usa a imagem inteira
    if n == 0:
        return 0, H, 0, W, np.ones_like(small, dtype=bool), scale
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    mask = ndi.binary_fill_holes(lab == sizes.argmax())
    
    # Remoção do peitoral: nos 35% superiores, apaga pixels muito claros (> 0.75)
    if cfg.get("REMOVE_PECTORAL", True):
        top_slice = mask[:int(mask.shape[0]*0.35), :]
        if top_slice.any():
            pectoral_mask = (small[:int(mask.shape[0]*0.35), :] > 0.75) & top_slice
            mask[:int(mask.shape[0]*0.35), :][pectoral_mask] = False

    # Caixa envolvente da mama (com margem de 2 px) na escala original
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, H, 0, W, np.ones_like(small, dtype=bool), scale
    mg = 2
    y0 = max(0, int((ys.min() - mg) / scale))
    y1 = min(H, int((ys.max() + mg + 1) / scale))
    x0 = max(0, int((xs.min() - mg) / scale))
    x1 = min(W, int((xs.max() + mg + 1) / scale))
    return y0, y1, x0, x1, mask, scale


# Guarda posições relativas (x, y dentro da caixa da mama) de todas as lesões conhecidas (pool empírico, não é KDE de fato)
def build_spatial_kde(df, cfg):
    """Extrai a lista empírica de posições relativas (cx_rel, cy_rel) das lesões anormais para pareamento 1:1."""
    # Só imagens que têm coordenada de lesão
    valid = df[df.roi_x.notna() & df.roi_y.notna()]
    if len(valid) == 0: return None
    
    pairs = []
    # Para cada lesão, normaliza a posição pela caixa da mama
    for r in valid.itertuples():
        arr = np.asarray(Image.open(r.path)).astype(np.float32)
        if arr.ndim == 3: arr = arr[..., :3].mean(-1)
        H, W = arr.shape
        img01 = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        geom = breast_geometry(img01, cfg)
        y0, y1, x0, x1 = geom[:4]
        
        cx_rel = (r.roi_x - x0) / max(x1 - x0, 1)
        cy_rel = (r.roi_y - y0) / max(y1 - y0, 1)
        pairs.append((cx_rel, cy_rel))
        
    return pairs


# Define a janela quadrada (top, left, lado) a recortar
def roi_window(shape, geom, roi_x, roi_y, idx, cfg, kde_model=None):
    H, W = shape
    y0, y1, x0, x1, mask_small, scale = geom
    # Lado da janela = fração da altura da mama (limitado ao tamanho da imagem)
    s = int(max(8, min(cfg["ROI_SIZE_FRAC"] * (y1 - y0), H, W)))
    # Distância de cada pixel até a borda da mama (para a janela caber dentro do tecido)
    cheb = ndi.distance_transform_cdt(np.pad(mask_small, 1), metric="chessboard")[1:-1, 1:-1]
    m = 2
    # Raio mínimo de distância à borda para a janela caber (na escala reduzida)
    r = int(np.ceil(s / 2 * scale)) + m
    # Mama pequena demais: reduz a janela
    if cheb.max() <= r:
        r = max(m + 1, int(cheb.max()) - 1)
        s = max(8, int(2 * (r - m) / scale))

    # Regra geral: anormal -> janela contém a lesão; normal -> janela em posição anatômica pareada
    # CASO ANORMAL: Centrado na lesão
    if np.isfinite(roi_x) and np.isfinite(roi_y):
        cx, cy = float(roi_x), float(roi_y)
        # Margem para a lesão não ficar colada na borda da janela
        mg = 0.15 * s
        # Imagem integral: calcula rápido a fração de tecido dentro de qualquer janela
        ii = np.pad(mask_small.astype(np.float32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        hs, ws = mask_small.shape

        # Fração da janela (t, l) coberta por tecido
        def cover(t, l):
            t0, l0 = min(int(t * scale), hs), min(int(l * scale), ws)
            t1, l1 = min(int(np.ceil((t + s) * scale)), hs), min(int(np.ceil((l + s) * scale)), ws)
            a = max((t1 - t0) * (l1 - l0), 1)
            return (ii[t1, l1] - ii[t0, l1] - ii[t1, l0] + ii[t0, l0]) / a

        # Faixa de posições da janela que ainda contêm a lesão com margem
        def span(c, size):
            lo, hi = max(0.0, c - s + mg), min(float(size - s), c - mg)
            if lo > hi:
                v = float(np.clip(c - s / 2, 0, size - s))
                return v, v
            return lo, hi

        # Faixas vertical e horizontal permitidas
        (tlo, thi), (llo, lhi) = span(cy, H), span(cx, W)
        # Testa grade 13x13 de posições; escolhe a com mais tecido, penalizando afastar a lesão do centro
        best = None
        for t in np.linspace(tlo, thi, 13):
            for l in np.linspace(llo, lhi, 13):
                off = max(abs(cy - (t + s / 2)), abs(cx - (l + s / 2))) / s
                sc = cover(t, l) - 0.10 * off
                if best is None or sc > best[0]:
                    best = (sc, t, l)
        return int(best[1]), int(best[2]), s

    # CASO NORMAL: Pareamento Empírico 1:1 com Projeção Anatômica Restrita
    # Gerador aleatório determinístico por imagem
    rng = np.random.RandomState(cfg["ROI_SEED"] + idx)
    
    if cfg.get("MATCH_SPATIAL_DIST", True) and kde_model is not None and len(kde_model) > 0:
        # Posições válidas: janela cabe inteira dentro da mama
        valid = cheb > r
        if not valid.any(): valid = cheb == cheb.max()
        valid_coords = np.argwhere(valid)
        
        # Avalia até 20 amostras do pool anormal para encontrar o melhor encaixe anatômico dentro do tecido
        best_pt = None
        min_dist = float("inf")
        
        # Sorteia até 20 lesões do pool como alvos
        sample_indices = rng.choice(len(kde_model), size=min(20, len(kde_model)), replace=False)
        for s_idx in sample_indices:
            cx_rel, cy_rel = kde_model[s_idx]
            # Converte a posição relativa da lesão para coordenadas desta mama
            target_cx = x0 + cx_rel * (x1 - x0)
            target_cy = y0 + cy_rel * (y1 - y0)
            
            # Converte para a escala da máscara reduzida
            py_target = int(np.clip(target_cy * scale, 0, mask_small.shape[0] - 1))
            px_target = int(np.clip(target_cx * scale, 0, mask_small.shape[1] - 1))
            
            # Ponto válido mais próximo do alvo
            dists = (valid_coords[:, 0] - py_target)**2 + (valid_coords[:, 1] - px_target)**2
            closest_idx = np.argmin(dists)
            curr_dist = dists[closest_idx]
            
            # Mantém o alvo mais bem encaixado entre os 20
            if curr_dist < min_dist:
                min_dist = curr_dist
                best_pt = valid_coords[closest_idx]
                
        # Sem candidato: sorteia posição válida; depois volta à escala original
        best_py, best_px = best_pt if best_pt is not None else valid_coords[rng.randint(len(valid_coords))]
        cy, cx = best_py / scale, best_px / scale
    else:
        # Sem pool de lesões: posição aleatória válida
        valid = cheb > r
        if not valid.any(): valid = cheb == cheb.max()
        valid_coords = np.argwhere(valid)
        py, px = valid_coords[rng.randint(len(valid_coords))]
        cy, cx = py / scale, px / scale
        
    # Converte o centro em canto superior esquerdo, limitado às bordas da imagem
    top = int(np.clip(cy - s / 2, 0, H - s))
    left = int(np.clip(cx - s / 2, 0, W - s))
    return top, left, s


# Processa uma imagem: recorte -> alinhamento -> normalização -> quadrado -> redimensionamento
def preprocess_one(path, roi_x, roi_y, idx, cfg, kde_model=None):
    # Lê em escala de cinza e normaliza para [0, 1]
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3: arr = arr[..., :3].mean(-1)
    H, W = arr.shape
    img01 = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Segmenta a mama e define a caixa de recorte
    geom = breast_geometry(img01, cfg)
    y0, y1, x0, x1 = geom[:4]
    crop01 = img01[y0:y1, x0:x1]
    # Se a metade direita for mais clara, a mama está do lado oposto: marca para espelhar
    cw = crop01.shape[1]
    flip = crop01[:, cw // 2:].mean() > crop01[:, :cw // 2].mean()

    # Recorta janela ROI ou mama inteira
    if cfg["ROI_MODE"] == "roi":
        top, left, s = roi_window((H, W), geom, roi_x, roi_y, idx, cfg, kde_model)
        out = arr[top:top + s, left:left + s]
    else:
        out = arr[y0:y1, x0:x1]
    # Espelha para todas as mamas ficarem do mesmo lado
    if flip:
        out = np.ascontiguousarray(out[:, ::-1])

    # Normalização de intensidade: CLAHE (padrão), percentis ou min-max
    norm_mode = cfg.get("NORM_MODE", "clahe")
    if norm_mode == "clahe":
        norm_temp = (out - out.min()) / (out.max() - out.min() + 1e-8)
        out = equalize_adapthist(norm_temp, clip_limit=0.02)
    elif norm_mode == "pct":
        lo, hi = np.percentile(out, cfg["NORM_PCT"])
        out = np.clip((out - lo) / (hi - lo + 1e-8), 0, 1)
    else:
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)

    # Completa com zeros para ficar quadrada
    if cfg["PAD_TO_SQUARE"]:
        h, w = out.shape
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=np.float32)
        canvas[:h, :w] = out
        out = canvas

    # Converte para uint8 e redimensiona para IMG_SIZE x IMG_SIZE
    img = Image.fromarray((out * 255).astype(np.uint8)).resize((cfg["IMG_SIZE"], cfg["IMG_SIZE"]), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


# Parâmetros que entram na chave do cache (mudou algum -> reprocessa)
PREPROC_KEYS = ["IMG_SIZE", "ROI_MODE", "ROI_SIZE_FRAC", "ROI_SEED", "MASK_MAX_SIDE",
                "BG_THRESH_FACTOR", "PAD_TO_SQUARE", "NORM_MODE", "NORM_PCT", "DDSM_SITES", 
                "MATCH_SPATIAL_DIST", "REMOVE_PECTORAL"]


# Gera (ou carrega do cache) o array de imagens de um dataset
def build_dataset(name, df):
    # Chave do cache = parâmetros + coordenadas + lista de arquivos
    cfg = {k: CONFIG[k] for k in PREPROC_KEYS}
    coords = hashlib.md5(df[["roi_x", "roi_y"]].round(1).to_numpy(dtype=float).tobytes()).hexdigest()
    key = json.dumps(cfg, sort_keys=True) + "|roi_v12_projected_pairmatching|" + coords + "|" + name + "|".join(df["path"].tolist())
    f = Path(CONFIG["CACHE_DIR"]) / f"{name}_{hashlib.md5(key.encode()).hexdigest()[:10]}.npy"
    
    # Cache existente: reutiliza
    if f.exists():
        print(f"[{name}] cache carregado: {f.name}")
        return np.load(f)
        
    t0 = time.time()
    # Pool de posições de lesões, usado para parear os normais
    kde_model = build_spatial_kde(df, cfg) if cfg.get("MATCH_SPATIAL_DIST") else None
    
    # Pré-processa todas as imagens em paralelo
    imgs = Parallel(n_jobs=CONFIG["N_JOBS"])(
        delayed(preprocess_one)(r.path, r.roi_x, r.roi_y, i, cfg, kde_model) for i, r in enumerate(df.itertuples()))
    # Empilha em array (N, IMG_SIZE, IMG_SIZE) e salva no cache
    X = np.stack(imgs)
    np.save(f, X)
    print(f"[{name}] {len(X)} imagens pré-processadas sem atalhos em {time.time() - t0:.0f}s -> {f.name}")
    return X


# Gera figuras de conferência: imagem original com caixas (vermelha = mama, verde/amarela = ROI) e resultado final
def show_samples(name, n_per_class=3, seed=0):
    df = load_df(name)
    kde_model = build_spatial_kde(df, CONFIG)
    picks = pd.concat([
        df[df.label == c].sample(min(n_per_class, (df.label == c).sum()), random_state=seed)
        for c in (0, 1)
    ])

    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 6))
    for k, r in enumerate(picks.itertuples()):
        arr = np.asarray(Image.open(r.path)).astype(np.float32)
        if arr.ndim == 3: arr = arr[..., :3].mean(-1)
        H, W = arr.shape
        img01 = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        geom = breast_geometry(img01, CONFIG)
        y0, y1, x0, x1 = geom[:4]

        ax = axes[0, k]
        ax.imshow(img01, cmap="gray")
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=2))
        if CONFIG["ROI_MODE"] == "roi":
            top, left, s = roi_window((H, W), geom, r.roi_x, r.roi_y, r.Index, CONFIG, kde_model)
            has_lesion = np.isfinite(r.roi_x) and np.isfinite(r.roi_y)
            ax.add_patch(plt.Rectangle((left, top), s, s, fill=False,
                                       ec="lime" if has_lesion else "yellow", lw=2))
        ax.set_title(f"{'Anormal' if r.label else 'Normal'}\n{Path(r.path).stem[:14]}", fontsize=8)

        axes[1, k].imshow(preprocess_one(r.path, r.roi_x, r.roi_y, r.Index, CONFIG, kde_model), cmap="gray")

    for ax in axes.ravel(): ax.axis("off")
    fig.suptitle(f"{name.upper()} — pareamento com projeção restrita | verde = lesão, amarelo = janela pareada", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"amostras_{name}.png", dpi=150)
    plt.close(fig)


# Monta a tabela final de um dataset, já com os filtros
def load_df(name):
    df = load_mias_metadata() if name == "mias" else derive_ddsm_roi(load_ddsm_metadata())
    # DDSM: filtra por site (letra inicial do nome do arquivo)
    if name == "ddsm" and CONFIG.get("DDSM_SITES"):
        site = df.path.map(lambda p: (re.match(r"^([A-Za-z])_\d+", Path(p).name) or [None, "?"])[1].upper())
        keep = site.isin([x.upper() for x in CONFIG["DDSM_SITES"]])
        print(f"[ddsm] DDSM_SITES={CONFIG['DDSM_SITES']}: mantendo {int(keep.sum())} de {len(df)} imagens")
        df = df[keep].reset_index(drop=True)
    # Descarta anormais sem coordenada da lesão
    if CONFIG["DROP_ABNORMAL_NO_ROI"]:
        bad = (df.label == 1) & (df.roi_x.isna() | df.roi_y.isna())
        print(f"[{name}] descartando {int(bad.sum())} anormais sem coordenada/máscara de {int((df.label == 1).sum())}"
              f" | restam {int((~bad).sum())} imagens (normais: {int((df.label == 0).sum())})")
        df = df[~bad].reset_index(drop=True)
    return df


# Execução: python gerar_dataset_roi.py  (ou --amostras para só gerar as figuras)
if __name__ == "__main__":
    # Confere se as pastas existem
    for name, key in [("mias", "MIAS_DIR"), ("ddsm", "DDSM_DIR")]:
        assert Path(CONFIG[key]).exists(), f"pasta não encontrada ({name}): {CONFIG[key]}"

    # Modo de conferência visual: gera figuras e sai
    if "--amostras" in sys.argv:
        for name in ["mias", "ddsm"]:
            show_samples(name)
        print("figuras salvas: amostras_mias.png, amostras_ddsm.png")
        sys.exit()

    # Para cada dataset: metadados -> pré-processamento -> arrays
    arrays, report = {}, []
    for name in ["mias", "ddsm"]:
        df = load_df(name)
        X = build_dataset(name, df)
        has_roi = df.roi_x.notna() & df.roi_y.notna()
        # X = imagens, y = rótulos, g = paciente (para dividir treino/teste por paciente)
        arrays[f"{name}_X"] = X
        arrays[f"{name}_y"] = df.label.values
        arrays[f"{name}_g"] = np.array(df.patient.tolist(), dtype=str)
        report.append(df.assign(dataset=name, has_roi=has_roi))
        print(f"[{name}] {len(df)} imagens | anormais: {int(df.label.sum())} | com coordenada da lesão: {int(has_roi.sum())}")

    # Relatório por imagem e dataset final comprimido
    pd.concat(report).to_csv("roi_report.csv", index=False)
    np.savez_compressed(CONFIG["OUT_NPZ"], **arrays)
    print(f"Salvo com sucesso: {CONFIG['OUT_NPZ']} | roi_report.csv")