"""Auditoria e conferência visual da ROI gerada por gerar_dataset.py

Coloque este arquivo na MESMA pasta do gerar_dataset.py (ele reutiliza o CONFIG e as funções de lá,
então o que você vê aqui é exatamente o que foi para o .npz).

Uso:
    python validar_img.py                      # roda tudo
    python validar_img.py --auditoria          # só números (rótulo sem coordenada, janelas suspeitas...)
    python validar_img.py --figuras            # só figuras sobre a imagem ORIGINAL (lesão x janela)
    python validar_img.py --npz                # só checagens do .npz (grade do que o modelo vê + testes de atalho)
    python validar_img.py --max-ddsm 800 --n 8 # mais imagens do DDSM / mais colunas por figura

Dependências extras: scikit-learn (opcional, só para os testes de atalho).

Saídas:
    diagnostico_mias.csv / diagnostico_ddsm.csv      uma linha por imagem, com flags
    fig_<dataset>_<categoria>.png                    original com bbox da mama (vermelho), janela (verde/amarelo),
                                                     lesão (+ ciano; círculo = raio no MIAS; magenta = máscara no DDSM)
    npz_<dataset>.png                                grade de 8 imagens por classe, direto do .npz
"""

import sys, argparse, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib, re
# usa a versão mais nova que estiver na pasta: v3 > v2 > original
_HERE = Path(__file__).resolve().parent
_GEN = next((n for n in ("gerar_dataset_v3", "gerar_dataset_v2") if (_HERE / f"{n}.py").exists()),
            "gerar_dataset")
g = importlib.import_module(_GEN)
print(f"[validar_img] usando {_GEN}.py")
CFG = g.CONFIG


# ---------------------------------------------------------------- utilidades
def load_gray(path):
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(-1)
    img01 = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr, img01


def find_mask(p):
    p = Path(p)
    for c in (p.with_name(f"{p.stem}_Mask.jpg"), p.with_name(f"Mask_{p.stem}.jpg")):
        if c.exists():
            return c
    return next((f for f in p.parent.iterdir()
                 if "mask" in f.name.lower() and p.stem.lower() in f.name.lower()), None)


def read_mask_bool(mp):
    m = np.asarray(Image.open(mp).convert("L")).astype(np.float32)
    if m.max() < 128:
        return None                      # máscara vazia
    m = m > 0.5 * m.max()
    if m.mean() > 0.5:
        m = ~m
    return m


def mias_radii():
    """raio da lesão (px) por imagem, só para desenhar o círculo na figura."""
    out = {}
    try:
        info = Path(CFG["MIAS_INFO"])
        if info.suffix.lower() == ".csv":
            tab = pd.read_csv(info)
            tab.columns = [str(c).strip().lower() for c in tab.columns]
            ref_col = next((c for c in ["refnum", "ref", "reference", "image", "filename", "file", "id", "name"]
                            if c in tab.columns), tab.columns[0])
            rc = next((c for c in ("radius", "rad", "r") if c in tab.columns), None)
            if rc:
                for _, r in tab.iterrows():
                    out.setdefault(Path(str(r[ref_col])).stem.lower(), pd.to_numeric(r[rc], errors="coerce"))
        else:
            for line in open(info, errors="ignore"):
                t = line.split()
                if t and t[0].lower().startswith("mdb") and len(t) >= 7:
                    out.setdefault(t[0].lower(), float(t[6]))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- diagnóstico por imagem
def diagnose(path, roi_x, roi_y, idx, kde_model=None):
    arr, img01 = load_gray(path)
    H, W = arr.shape
    geom = g.breast_geometry(img01, CFG)
    y0, y1, x0, x1, mask, scale = geom
    top, left, s = g.roi_window((H, W), geom, roi_x, roi_y, idx, CFG, kde_model=kde_model)

    cheb = ndi.distance_transform_cdt(np.pad(mask, 1), metric="chessboard")[1:-1, 1:-1]
    cy_w, cx_w = top + s / 2, left + s / 2
    py = int(np.clip(cy_w * scale, 0, mask.shape[0] - 1))
    px = int(np.clip(cx_w * scale, 0, mask.shape[1] - 1))
    t, b = int(top * scale), int(np.ceil((top + s) * scale))
    l, r = int(left * scale), int(np.ceil((left + s) * scale))
    win = img01[top:top + s, left:left + s]

    d = dict(H=H, W=W,
             bbox_frac=(y1 - y0) * (x1 - x0) / (H * W),          # ~1.0 => máscara da mama provavelmente falhou
             win_side=s, win_frac_breast=s / max(y1 - y0, 1),
             top=top, left=left,
             cx_rel=(cx_w - x0) / max(x1 - x0, 1), cy_rel=(cy_w - y0) / max(y1 - y0, 1),
             edge_dist=cheb[py, px] / scale / s,                  # distância à borda da mama, em lados de janela
             cover=mask[t:b, l:r].mean() if (b > t and r > l) else np.nan,
             win_mean=float(win.mean()), win_std=float(win.std()), win_dark=float((win < 0.05).mean()),
             lesion_in_win=np.nan, lesion_off=np.nan)
    if np.isfinite(roi_x) and np.isfinite(roi_y):
        d["lesion_in_win"] = bool(left <= roi_x < left + s and top <= roi_y < top + s)
        d["lesion_off"] = float(max(abs(roi_x - cx_w), abs(roi_y - cy_w)) / s)   # 0 = centrada, 0.5 = na borda
    return d


def run_diag(df):
    kde_model = g.build_spatial_kde(df, CFG) if CFG.get("MATCH_SPATIAL_DIST") and hasattr(g, "build_spatial_kde") else None
    rows = Parallel(n_jobs=CFG["N_JOBS"])(
        delayed(diagnose)(r.path, r.roi_x, r.roi_y, r.Index, kde_model=kde_model) for r in df.itertuples())
    return pd.concat([df.reset_index().rename(columns={"index": "idx"}), pd.DataFrame(rows)], axis=1)


def sample_df(df, n_total, seed=0):
    if n_total is None or len(df) <= n_total:
        return df
    parts = [df[df.label == c].sample(min((df.label == c).sum(), n_total // 2), random_state=seed) for c in (0, 1)]
    return pd.concat(parts).sort_index()


# ---------------------------------------------------------------- testes de atalho (sklearn)
def cv_auc(F, y, groups, model="hgb"):
    try:
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return None
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
        return None
    clf = (HistGradientBoostingClassifier(max_depth=3, max_iter=150) if model == "hgb"
           else make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)))
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    p = cross_val_predict(clf, np.nan_to_num(F), y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


# ---------------------------------------------------------------- auditoria
def audit(name, df_full, diag):
    ab = df_full[df_full.label == 1]
    has = ab.roi_x.notna() & ab.roi_y.notna()
    print(f"\n=== {name.upper()} ===")
    print(f"imagens: {len(df_full)} | normais: {(df_full.label == 0).sum()} | anormais: {len(ab)} | pacientes: {df_full.patient.nunique()}")
    print(f"anormais COM coordenada/máscara : {has.sum()}")
    print(f"anormais SEM coordenada         : {(~has).sum()}  ({100 * (~has).mean():.0f}% das anormais)"
          "  <- recorte ALEATÓRIO com rótulo 'anormal' = ruído de rótulo")
    d = diag
    print(f"\n(diagnóstico geométrico em {len(d)} imagens)")
    print(f"  máscara da mama suspeita (bbox > 90% da imagem): {(d.bbox_frac > 0.9).sum()}")
    print(f"  janela com >5% de pixels pretos (<0.05)         : {(d.win_dark > 0.05).sum()}")
    wl = d[d.lesion_in_win.notna()]
    if len(wl):
        print(f"  lesão FORA da janela                             : {(wl.lesion_in_win == False).sum()} de {len(wl)}")
        print(f"  lesão deslocada do centro > 0.25 lado (janela empurrada p/ dentro da mama): {(wl.lesion_off > 0.25).sum()} de {len(wl)}")
        print(f"  lesão deslocada > 0.40 (quase na borda da janela): {(wl.lesion_off > 0.40).sum()}")
    print(f"  lado da janela (px orig.): mediana {d.win_side.median():.0f} | min {d.win_side.min()} | max {d.win_side.max()}"
          f"  -> reduzida p/ {CFG['IMG_SIZE']} (fator mediano {d.win_side.median() / CFG['IMG_SIZE']:.1f}x)")

    print("\n  janelas com fundo (comparar normais x anormais: a diferença É um atalho):")
    for lab, nm in ((0, "normais"), (1, "anormais")):
        sub = d[d.label == lab]
        if len(sub):
            print(f"    {nm:9s}: <90% dentro da mama: {(sub.cover < 0.9).sum():4d}/{len(sub)}"
                  f" | >5% pixels pretos: {(sub.win_dark > 0.05).sum():4d}/{len(sub)}"
                  f" | dist. mediana à borda: {sub.edge_dist.median():.2f} lados")
    if name == "ddsm":
        def site_of(pth):
            mt = re.match(r"^([A-Za-z])_\d+", Path(pth).name)
            return mt.group(1).upper() if mt else "?"
        ct = pd.crosstab(df_full.path.map(site_of), df_full.label)
        ct.columns = ["normal", "anormal"] if list(ct.columns) == [0, 1] else ct.columns
        if "anormal" in ct and "normal" in ct:
            ct["%anormal"] = (100 * ct["anormal"] / (ct["anormal"] + ct["normal"])).round(0)
        print("\n  ORIGEM (letra do caso DDSM) x classe  <- se uma letra existe em só uma classe, o modelo pode aprender a origem:")
        print("\n".join("    " + l for l in ct.to_string().splitlines()))
        print("   (para restringir: DDSM_SITES = [\"A\", \"D\"] no CONFIG do gerador)")

    # a posição da janela sozinha prevê o rótulo? (atalho geométrico)
    feats = ["cx_rel", "cy_rel", "edge_dist"]
    auc = cv_auc(d[feats].values, d.label.values, d.patient.values)
    if auc is not None:
        print(f"\n  TESTE DE ATALHO (posição da janela: x, y, dist. à borda) -> AUC por paciente = {auc:.2f}")
        print("   ~0.50 = ok | >0.65 = a POSIÇÃO do recorte entrega o rótulo (modelo pode aprender 'onde' e não 'o quê')")
    d.to_csv(f"diagnostico_{name}.csv", index=False)
    print(f"  salvo: diagnostico_{name}.csv")


# ---------------------------------------------------------------- figuras sobre a imagem original
def overlay_figure(name, rows, fname, title, radii, kde_model=None):
    n = len(rows)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.4), squeeze=False)
    for k, r in enumerate(rows.itertuples()):
        arr, img01 = load_gray(r.path)
        H, W = arr.shape
        geom = g.breast_geometry(img01, CFG)
        y0, y1, x0, x1 = geom[:4]
        top, left, s = g.roi_window((H, W), geom, r.roi_x, r.roi_y, r.Index, CFG, kde_model=kde_model)
        has = np.isfinite(r.roi_x) and np.isfinite(r.roi_y)

        st = max(1, max(H, W) // 1200)
        ax = axes[0, k]
        ax.imshow(img01[::st, ::st], cmap="gray", extent=(0, W, H, 0))
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.5))
        ax.add_patch(plt.Rectangle((left, top), s, s, fill=False, ec="lime" if has else "yellow", lw=2))
        note = ""
        if has:
            ax.plot(r.roi_x, r.roi_y, "+", color="cyan", ms=16, mew=2)
            rad = radii.get(Path(r.path).stem.lower()) if name == "mias" else None
            if rad is not None and np.isfinite(rad):
                ax.add_patch(plt.Circle((r.roi_x, r.roi_y), rad, fill=False, ec="cyan", ls="--", lw=1.5))
            if name == "ddsm" and r.label == 1:
                mp = find_mask(r.path)
                mm = read_mask_bool(mp) if mp is not None else None
                if mm is not None:
                    if hasattr(g, "lesion_region"):          # região preenchida usada para achar o centro
                        reg, fct = g.lesion_region(mm)
                        X = (np.arange(reg.shape[1]) + 0.5) * fct * W / mm.shape[1]
                        Y = (np.arange(reg.shape[0]) + 0.5) * fct * H / mm.shape[0]
                        mm_draw = reg
                    else:
                        mm_draw = np.asarray(Image.fromarray(mm.astype(np.uint8) * 255).resize((W, H), Image.NEAREST))[::st, ::st] > 0
                        X, Y = np.linspace(0, W, mm_draw.shape[1]), np.linspace(0, H, mm_draw.shape[0])
                    if mm_draw.any() and (~mm_draw).any():
                        ax.contour(X, Y, mm_draw.astype(float), levels=[0.5], colors="magenta", linewidths=1.2)
            inside = left <= r.roi_x < left + s and top <= r.roi_y < top + s
            note = "" if inside else " | LESÃO FORA!"
        elif r.label == 1:
            note = " | SEM coord (aleatória)"
        ax.set_title(f"{'Anormal' if r.label else 'Normal'}{note}\n{Path(r.path).stem[:18]}", fontsize=7)
        ax.axis("off")

        axes[1, k].imshow(g.preprocess_one(r.path, r.roi_x, r.roi_y, r.Index, CFG, kde_model=kde_model), cmap="gray", vmin=0, vmax=255)
        axes[1, k].set_title("o que vai pro .npz", fontsize=7)
        axes[1, k].axis("off")
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  figura: {fname}")


def figures(name, df, n, radii):
    kde_model = g.build_spatial_kde(df, CFG) if CFG.get("MATCH_SPATIAL_DIST") and hasattr(g, "build_spatial_kde") else None
    has = df.roi_x.notna() & df.roi_y.notna()
    cats = {
        "normais": df[df.label == 0],
        "anormais_com_coord": df[(df.label == 1) & has],
        "anormais_SEM_coord": df[(df.label == 1) & ~has],
    }
    for cat, sub in cats.items():
        if len(sub) == 0:
            continue
        pick = sub.sample(min(n, len(sub)), random_state=1)
        overlay_figure(name, pick, f"fig_{name}_{cat}.png",
                       f"{name.upper()} — {cat} | vermelho = mama | verde = janela c/ lesão | amarelo = janela sorteada "
                       f"| + ciano = lesão | magenta = máscara DDSM", radii, kde_model=kde_model)


# ---------------------------------------------------------------- checagens do .npz
def npz_checks(path):
    if not Path(path).exists():
        print(f"[npz] não encontrei {path}")
        return
    z = np.load(path, allow_pickle=False)
    for name in ("mias", "ddsm"):
        if f"{name}_X" not in z:
            continue
        X, y, grp = z[f"{name}_X"], z[f"{name}_y"], z[f"{name}_g"]
        print(f"\n=== NPZ {name.upper()} === X{X.shape} {X.dtype} | y: {dict(zip(*np.unique(y, return_counts=True)))}"
              f" | pacientes: {len(np.unique(grp))}")
        assert len(X) == len(y) == len(grp), "tamanhos diferentes!"

        # duplicatas exatas
        h = [hashlib.md5(x.tobytes()).hexdigest() for x in X]
        dup = len(h) - len(set(h))
        print(f"  imagens idênticas (hash): {dup}")

        # estatísticas por classe + features de intensidade
        feats = []
        for i in range(0, len(X), 256):
            c = X[i:i + 256].astype(np.float32)
            feats.append(np.stack([c.mean((1, 2)), c.std((1, 2)),
                                   np.percentile(c, 1, axis=(1, 2)), np.percentile(c, 99, axis=(1, 2)),
                                   (c <= 5).mean((1, 2)), (c >= 250).mean((1, 2)),
                                   np.abs(np.diff(c, axis=2)).mean((1, 2))], 1))
        F = np.concatenate(feats)
        names = ["média", "desvio", "p1", "p99", "%escuro", "%saturado", "grad.médio"]
        tab = pd.DataFrame(F, columns=names).assign(classe=y).groupby("classe").mean().round(2)
        print("  médias por classe:\n" + "\n".join("    " + l for l in tab.to_string().splitlines()))

        a1 = cv_auc(F, y, grp, "hgb")
        a2 = cv_auc(X.reshape(len(X), 14, 16, 14, 16).mean((2, 4)).reshape(len(X), -1), y, grp, "lr")
        if a1 is not None:
            print(f"  TESTE DE ATALHO (só estatísticas de intensidade)  AUC por paciente = {a1:.2f}")
            print(f"  BASELINE trivial (imagem 14x14 + regressão logística) AUC por paciente = {a2:.2f}")
            print("   -> se o baseline trivial já chega perto do seu CNN/Swin, o problema está nos DADOS/rótulos,")
            print("      não na arquitetura. Se as estatísticas de intensidade sozinhas dão AUC alto, há atalho.")

        # grade
        rng = np.random.RandomState(0)
        n = 8
        fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.4), squeeze=False)
        for row, c in enumerate((0, 1)):
            ids = np.where(y == c)[0]
            ids = rng.choice(ids, min(n, len(ids)), replace=False)
            for k in range(n):
                ax = axes[row, k]
                ax.axis("off")
                if k < len(ids):
                    ax.imshow(X[ids[k]], cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"{'anormal' if c else 'normal'} #{ids[k]}", fontsize=7)
        fig.suptitle(f"{name.upper()} — direto do .npz (o que o modelo recebe)", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"npz_{name}.png", dpi=140)
        plt.close(fig)
        print(f"  figura: npz_{name}.png")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auditoria", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    ap.add_argument("--npz", nargs="?", const=CFG["OUT_NPZ"], default=None)
    ap.add_argument("--max-ddsm", type=int, default=400, help="nº de imagens do DDSM no diagnóstico (balanceado por classe)")
    ap.add_argument("--n", type=int, default=6, help="colunas por figura")
    a = ap.parse_args()
    run_all = not (a.auditoria or a.figuras or a.npz)
    do_aud, do_fig = a.auditoria or run_all, a.figuras or run_all
    npz_path = a.npz or (CFG["OUT_NPZ"] if run_all else None)

    if do_aud or do_fig:
        radii = mias_radii()
        for name in ("mias", "ddsm"):
            if not Path(CFG[f"{name.upper()}_DIR"]).exists():
                print(f"[aviso] pasta de {name} não encontrada; pulando.")
                continue
            df = g.load_df(name)
            sub = sample_df(df, a.max_ddsm if name == "ddsm" else None)
            if do_aud:
                audit(name, df, run_diag(sub))
            if do_fig:
                figures(name, sub, a.n, radii)
    if npz_path:
        npz_checks(npz_path)


if __name__ == "__main__":
    main()