# %% [markdown]
# # Zanzibar clove grading: mechanism tests for the session confound for the Smart Agricultural Technology resubmission
#
# Run on Kaggle with **GPU T4 x1** and **Internet ON** (for ImageNet weights), via
# **Save Version -> Save & Run All (Commit)** so it runs in the background (12 h limit; expected ~5-6 h).
#
# Sections (each saves its own JSON to `/kaggle/working/results/`, and is skipped on rerun if already done):
# 0. Config and data loading
# 1. Dataset audit: counts, exact and near-duplicate images, duplicate-aware grouping
# 2. 18-architecture benchmark on one fixed grouped, stratified 70/15/15 split, identical protocol for all,
#    with bootstrap 95% CIs and exact McNemar tests (Holm-corrected) against the best model
# 3. Classical baseline: SVM on LAB colour statistics + GLCM texture
# 4. 10-fold grouped CV: monolithic RGB vs masked RGB vs decomposed RGB+mask ResNet-18, all retrained per fold
# 5. Grad-CAM: share of attention outside the clove mask (monolithic vs decomposed)
# 6. CPU latency (batch size 1, single thread) as an edge-device proxy
# 7. Batch-level ZSTC grading on the batch images (only if BATCH_DIR exists)
# 8. Summary tables + results.zip  -> send me results.zip and the notebook log

# %%
import os, sys, json, time, glob, math, hashlib, random, copy, zipfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from scipy.stats import binomtest, ttest_rel, wilcoxon, t as tdist

TEST_MODE = os.environ.get("CLOVE_TEST", "0") == "1"   # local smoke test only; leave unset on Kaggle

# ------------------------------------------------------------------ CONFIG (edit paths if needed)
def _find_dir(names, root="/kaggle/input"):
    """Return the first folder under root that contains all `names` as subfolders."""
    for dp, dn, _ in os.walk(root):
        if all(n in dn for n in names): return dp
    return None
def _find_ckpt(root="/kaggle/input"):
    c = sorted(glob.glob(os.path.join(root, "**", "*.pt*"), recursive=True))
    seg = [p for p in c if "seg" in os.path.basename(p).lower() or "unet" in p.lower()]
    return (seg or [None])[0], c

GRADES    = ["Grade_1", "Grade_2", "Grade_3", "Grade_4"]          # single-clove folders
BATCHES   = ["Group_Grade_1", "Group_Grade_2", "Group_Grade_3", "Group_Grade_4"]  # batch-photo folders
_ALT = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]
DATA_DIR  = os.environ.get("CLOVE_DATA") or _find_dir(GRADES) or _find_dir(_ALT)
if DATA_DIR and not os.path.isdir(os.path.join(DATA_DIR, GRADES[0])): GRADES = _ALT
BATCH_DIR = os.environ.get("CLOVE_BATCH") or _find_dir(BATCHES) or "none"
_ck, _all = _find_ckpt() if not TEST_MODE else (None, [])
UNET_CKPT = os.environ.get("CLOVE_UNET") or _ck or "none"
print("DATA_DIR :", DATA_DIR); print("BATCH_DIR:", BATCH_DIR); print("UNET_CKPT:", UNET_CKPT)
if not TEST_MODE:
    assert DATA_DIR, f"No folder with {GRADES} under /kaggle/input. Attach the dataset (Add Input)."
    if UNET_CKPT == "none":
        print("WARNING: no U-Net checkpoint attached -> Stage 1 uses Otsu thresholding (as in the paper's pre-processing).")
    import urllib.request
    try: urllib.request.urlopen("https://huggingface.co", timeout=10)
    except Exception: raise SystemExit("No internet: turn Internet ON in the notebook settings (needed for ImageNet weights).")
OUT       = os.environ.get("CLOVE_OUT", "/kaggle/working/results_session")
SEED      = 42
IMG       = int(os.environ.get("CLOVE_IMG", 64 if TEST_MODE else 224))
MAX_EPOCHS, PATIENCE, BS, LR, WD = (2, 1, 8, 1e-4, 0.05) if TEST_MODE else (60, 8, 32, 1e-4, 0.05)
N_FOLDS   = 2 if TEST_MODE else 10
N_BOOT    = 200 if TEST_MODE else 2000
NEAR_DUP_HAMMING = 4      # dHash 64-bit: <=4 differing bits treated as the same physical clove/shot
PRETRAINED = not TEST_MODE
R18_ID = "resnet18.tv_in1k" if PRETRAINED else "resnet18"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = dev.type == "cuda"

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)

def save(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w") as f: json.dump(obj, f, indent=2, default=float)
def done(name): return os.path.exists(os.path.join(OUT, name + ".json"))
def load(name):
    with open(os.path.join(OUT, name + ".json")) as f: return json.load(f)

env = {"torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
       "python": sys.version.split()[0], "seed": SEED, "img": IMG, "max_epochs": MAX_EPOCHS,
       "patience": PATIENCE, "batch_size": BS, "lr": LR, "weight_decay": WD}
try:
    import timm; env["timm"] = timm.__version__
except ImportError:
    raise SystemExit("timm missing: run `pip install timm` in a cell above")
print(env); save("00_environment", env)

# %% [markdown]
# ## 0. Load all single-clove images into RAM (uint8) once

# %%
paths, labels = [], []
for i, g in enumerate(GRADES):
    d = os.path.join(DATA_DIR, g)
    assert os.path.isdir(d), f"Missing folder {d}"
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(d, f)); labels.append(i)
paths, labels = np.array(paths), np.array(labels)
N = len(paths)
orig_sizes = []
X = np.zeros((N, IMG, IMG, 3), np.uint8)
for k, p in enumerate(paths):
    im = Image.open(p).convert("RGB"); orig_sizes.append(im.size)
    X[k] = np.asarray(im.resize((IMG, IMG), Image.BILINEAR))
X_t = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()   # N,3,H,W uint8 (CPU)
Y_t = torch.from_numpy(labels).long()
print("Images per grade:", dict(zip(GRADES, np.bincount(labels, minlength=4).tolist())), "total", N)

# %% [markdown]
# ## 1. Dataset audit and duplicate-aware grouping
# Near-identical photos of the same clove in both train and test inflate scores. We hash every image,
# cluster near-duplicates, and use the clusters as groups so no cluster is split across train/test.

# %%
def dhash(arr):
    g = np.asarray(Image.fromarray(arr).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())  # 8 bytes

md5 = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in paths]
H = np.stack([dhash(X[k]) for k in range(N)])                        # N x 8 uint8
bits = np.unpackbits(H, axis=1).astype(np.uint8)                     # N x 64
parent = list(range(N))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
near_pairs, cross_grade_pairs = 0, 0
for s in range(0, N, 512):
    blk = bits[s:s + 512]
    ham = (blk[:, None, :] != bits[None, :, :]).sum(2)               # 512 x N
    ii, jj = np.where(ham <= NEAR_DUP_HAMMING)
    for a, b in zip(ii + s, jj):
        if a < b:
            near_pairs += 1
            if labels[a] != labels[b]: cross_grade_pairs += 1
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb
groups = np.array([find(k) for k in range(N)])
sizes = np.bincount(np.unique(groups, return_inverse=True)[1])
audit = {
    "n_images": N, "per_grade": np.bincount(labels, minlength=4).tolist(),
    "original_sizes_top5": {f"{w}x{h}": c for (w, h), c in
                            sorted({s: orig_sizes.count(s) for s in set(orig_sizes)}.items(), key=lambda x: -x[1])[:5]},
    "exact_duplicate_files": int(N - len(set(md5))),
    "near_duplicate_pairs": int(near_pairs), "near_duplicate_pairs_with_different_grade": int(cross_grade_pairs),
    "n_groups": int(len(sizes)), "largest_group": int(sizes.max()), "images_in_multi_image_groups": int(sizes[sizes > 1].sum()),
    "hamming_threshold": NEAR_DUP_HAMMING,
}
print(json.dumps(audit, indent=2)); save("01_audit", audit)

# %% [markdown]
# ## Shared training utilities (GPU augmentation, AMP, early stopping on validation macro-F1)

# %%
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def to_input(xb_u8, mb=None, mode="rgb"):
    """xb_u8: B,3,H,W uint8 on device; mb: B,1,H,W float mask in [0,1]."""
    x = (xb_u8.float() / 255 - MEAN.to(xb_u8.device)) / STD.to(xb_u8.device)
    if mode == "rgb":    return x
    if mode == "masked": return x * (mb > 0.5).float()               # background zeroed (post-normalisation)
    if mode == "rgbm":   return torch.cat([x, mb * 2 - 1], 1)        # 4th channel scaled to [-1,1]
    raise ValueError(mode)

def augment(x):
    """Same augmentation for every model: h/v flip, rotation +-15 deg, brightness +-20% (RGB channels only)."""
    B = x.shape[0]
    f = torch.rand(B, device=x.device) < 0.5; x[f] = x[f].flip(3)
    f = torch.rand(B, device=x.device) < 0.5; x[f] = x[f].flip(2)
    ang = (torch.rand(B, device=x.device) * 30 - 15) * math.pi / 180
    cos, sin = torch.cos(ang), torch.sin(ang); z = torch.zeros_like(cos)
    theta = torch.stack([torch.stack([cos, -sin, z], 1), torch.stack([sin, cos, z], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    x[:, :3] = x[:, :3] * (0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=x.device))
    return x

def make_model(name, in_ch=3):
    import timm
    m = timm.create_model(name, pretrained=PRETRAINED, num_classes=4, in_chans=in_ch)  # timm adapts conv1 for in_ch=4
    return m.to(dev)

def predict(model, idx, mode, masks=None, bs=64):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(dev.type, enabled=AMP):
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            xb = X_t[b].to(dev); mb = masks[b].to(dev) if masks is not None else None
            out.append(model(to_input(xb, mb, mode)).float().softmax(1).cpu())
    return torch.cat(out).numpy()

def train_eval(name, tr, va, te, mode="rgb", masks=None, seed=SEED, return_model=False):
    seed_all(seed)
    model = make_model(name, 4 if mode == "rgbm" else 3)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    scaler = torch.amp.GradScaler(dev.type, enabled=AMP)
    best, best_state, bad, best_ep, t0 = -1.0, None, 0, 0, time.time()
    for ep in range(MAX_EPOCHS):
        model.train(); perm = np.random.permutation(tr)
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            if len(b) < 2: continue
            xb = X_t[b].to(dev, non_blocking=True); mb = masks[b].to(dev) if masks is not None else None
            yb = Y_t[b].to(dev)
            with torch.autocast(dev.type, enabled=AMP):
                loss = F.cross_entropy(model(augment(to_input(xb, mb, mode))), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        vf = f1_score(labels[va], predict(model, va, mode, masks).argmax(1), average="macro")
        if vf > best: best, best_state, bad, best_ep = vf, copy.deepcopy(model.state_dict()), 0, ep + 1
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    prob = predict(model, te, mode, masks)
    res = {"val_macro_f1": best, "best_epoch": best_ep, "epochs_run": ep + 1, "train_minutes": (time.time() - t0) / 60,
           "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    return (res, prob, model) if return_model else (res, prob)

def boot_ci(y, p, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v.append(f1_score(y[i], p[i], average="macro"))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def mcnemar_exact(y, pa, pb):
    b = int(((pa == y) & (pb != y)).sum()); c = int(((pa != y) & (pb == y)).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"a_only_correct": b, "b_only_correct": c, "p": float(p)}

def holm(pvals):
    order = np.argsort(pvals); adj = np.empty(len(pvals)); m = len(pvals); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i])); adj[i] = run
    return adj.tolist()

def metrics(y, pred):
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], zero_division=0)
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1_ci95": boot_ci(y, pred), "per_class": {"precision": pr.tolist(), "recall": rc.tolist(),
            "f1": f1.tolist(), "support": sup.tolist()}, "confusion": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist()}

# %% [markdown]
# ## Fixed grouped, stratified 70/15/15 split (used by sections 2, 3 and 6)

# %%
if done("02_split"):
    sp = load("02_split"); tr_idx, va_idx, te_idx = map(np.array, (sp["train"], sp["val"], sp["test"]))
else:
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)          # ~1/7 test
    rest, te_idx = next(sgkf.split(paths, labels, groups))
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)         # ~1/6 of rest -> val
    a, b = next(sgkf2.split(paths[rest], labels[rest], groups[rest]))
    tr_idx, va_idx = rest[a], rest[b]
    save("02_split", {"train": tr_idx.tolist(), "val": va_idx.tolist(), "test": te_idx.tolist(),
                      "train_files": [os.path.relpath(p, DATA_DIR) for p in paths[tr_idx]],
                      "val_files": [os.path.relpath(p, DATA_DIR) for p in paths[va_idx]],
                      "test_files": [os.path.relpath(p, DATA_DIR) for p in paths[te_idx]]})
assert not (set(groups[tr_idx]) & set(groups[te_idx])) and not (set(groups[va_idx]) & set(groups[te_idx]))
for nm, ix in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
    print(nm, len(ix), np.bincount(labels[ix], minlength=4).tolist())

# %% [markdown]
# ## U-Net masks for all images (needed by sections 4, 5, 7)
# The U-Net checkpoint is reused, not retrained per fold. Report this; if pixel masks exist, retraining it per fold is better.

# %%
class EnhancedDoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                  nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(i, o, 1, bias=False) if i != o else nn.Identity()
    def forward(self, x): return self.relu(self.conv(x) + self.residual(x))

class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2); c = in_channels
        for f in features: self.downs.append(EnhancedDoubleConv(c, f)); c = f
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2)); self.ups.append(EnhancedDoubleConv(f * 2, f))
        self.bottleneck = EnhancedDoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1); self.dropout = nn.Dropout2d(0.2)
    def forward(self, x):
        skips = []
        for d in self.downs: x = d(x); skips.append(x); x = self.pool(x)
        x = self.dropout(self.bottleneck(x)); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i // 2]
            if x.shape != s.shape: x = F.interpolate(x, size=s.shape[2:])
            x = self.ups[i + 1](torch.cat((s, x), 1))
        return torch.sigmoid(self.final_conv(x))

import cv2
USE_UNET = os.path.exists(UNET_CKPT)
unet = EnhancedUNet().to(dev).eval()
if USE_UNET:
    ck = torch.load(UNET_CKPT, map_location=dev)
    if isinstance(ck, dict) and "model_state_dict" in ck: ck = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck: ck = ck["state_dict"]
    unet.load_state_dict(ck)

def otsu_mask(rgb):
    """Otsu on grey + clean-up; polarity chosen so the image border is background."""
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]) > 0
    if border.mean() > 0.5: m = 255 - m          # background darker than cloves: flip so the clove is foreground
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return (m > 0).astype(np.float32)

def segment(rgb):
    """rgb: HxWx3 uint8 -> HxW float mask in {0,1}-ish."""
    if USE_UNET:
        with torch.no_grad():
            return unet(torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(dev) / 255)[0, 0].cpu().numpy()
    return otsu_mask(rgb)

M_t = torch.from_numpy(np.stack([segment(X[k]) for k in range(N)]))[:, None].float()
print("Stage 1 segmentation:", "U-Net checkpoint" if USE_UNET else "Otsu thresholding")
fg = (M_t > 0.5).float().mean((1, 2, 3)).numpy()
save("04_mask_stats", {"foreground_fraction_mean": float(fg.mean()), "foreground_fraction_p5_p95": np.percentile(fg, [5, 95]).tolist(),
                       "images_with_empty_mask": int((fg < 0.01).sum()),
                       "method": "U-Net checkpoint" if USE_UNET else "Otsu thresholding"})
print(load("04_mask_stats"))

# %% [markdown]
# ## 11. Session tests using capture dates (decisive tests requested by reviewers)
# Capture dates come from the filenames (YYYYMMDD_hhmmss.jpg). Grades share some dates, so we can:
# A. grade-by-date table; B. same-day grading (session cannot help); C. leave-date-out evaluation;
# D. silhouette controls (inpainted background, silhouette only, shape features);
# E. swap test with a paste-back control, repeated pairings, binomial test.
# Each block saves 11_*.json and is skipped on rerun.

# %%
import re
from scipy.stats import binomtest as _binom
from skimage.measure import regionprops, label as sklabel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEEDS = [42] if TEST_MODE else [42, 43, 44]
R18_ID = "resnet18.tv_in1k" if PRETRAINED else "resnet18"
EB0_ID = "efficientnet_b0.ra_in1k" if PRETRAINED else "efficientnet_b0"
X_orig = X_t.clone()

def date_of(p):
    m = re.search(r"(20\d{6})", os.path.basename(p))
    return m.group(1) if m else "unknown"
dates = np.array([date_of(p) for p in paths])
tab = {}
for g in range(4):
    u, c = np.unique(dates[labels == g], return_counts=True)
    tab[GRADES[g]] = dict(zip(u.tolist(), c.tolist()))
save("11_A_dates", tab); print(json.dumps(tab, indent=1))

def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - h) / d), float((c + h) / d)]

def f1_present(y, p):
    return float(f1_score(y, p, labels=sorted(set(y.tolist())), average="macro"))

def summarize(y, p):
    acc_k = int((y == p).sum())
    return {"n": int(len(y)), "macro_f1": f1_present(y, p), "accuracy": acc_k / len(y), "accuracy_wilson95": wilson(acc_k, len(y)),
            "confusion": confusion_matrix(y, p, labels=sorted(set(y.tolist()))).tolist()}

def run_subset_cv(idx, tensor, model_id, seed, nf):
    """Grouped stratified CV restricted to image indices idx, training on `tensor` images."""
    global X_t
    keep = X_t; X_t = tensor
    try:
        y = labels[idx]; preds = np.full(len(idx), -1)
        folds = list(StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=seed).split(idx, y, groups[idx]))
        for k, (trv, te) in enumerate(folds):
            a, b = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + k).split(idx[trv], y[trv], groups[idx[trv]]))
            res, prob = train_eval(model_id, idx[trv][a], idx[trv][b], idx[te], "rgb", seed=seed + k)
            preds[te] = prob.argmax(1)
        return summarize(y, preds)
    finally:
        X_t = keep

# ---------- background variants needed below
fgm = (M_t[:, 0].numpy() > 0.5).astype(np.uint8)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)); k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
tight = np.stack([cv2.dilate(fgm[i], k3) for i in range(N)]).astype(bool)
wide = np.stack([cv2.dilate(fgm[i], k15) for i in range(N)]).astype(np.uint8)
def inpaint(i):
    return cv2.inpaint(X[i], wide[i] * 255, 7, cv2.INPAINT_TELEA) if wide[i].any() else X[i].copy()
INP = np.stack([inpaint(i) for i in range(N)])                     # bud removed, silhouette not preserved
SIL = np.stack([np.repeat((fgm[i] * 255)[..., None], 3, 2) for i in range(N)]).astype(np.uint8)   # silhouette only
to_t = lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).contiguous()
INP_t, SIL_t = to_t(INP), to_t(SIL)

# %% [markdown]
# ### B. Same-day grading: only dates that contain two or more grades

# %%
NF = 2 if TEST_MODE else 5
B = load("11_B_sameday") if done("11_B_sameday") else {}
multi = [d for d in np.unique(dates) if len(np.unique(labels[dates == d])) >= 2 and d != "unknown"]
for d in multi:
    idx = np.where(dates == d)[0]
    if min(np.bincount(labels[idx])[np.unique(labels[idx])]) < NF * 2: continue
    for variant, T in [("original", X_orig), ("background_inpainted", INP_t)]:
        for mid, mname in [(R18_ID, "ResNet-18")]:
            for sd in SEEDS:
                key = f"{d}|{variant}|{mname}|{sd}"
                if key in B: continue
                B[key] = run_subset_cv(idx, T, mid, sd, NF); save("11_B_sameday", B)
                print("same-day", key, round(B[key]["macro_f1"] * 100, 2), flush=True)

# %% [markdown]
# ### C. Leave-date-out: train on some dates, test on dates never seen in training

# %%
C = load("11_C_leavedate") if done("11_C_leavedate") else {}
gdates = {g: sorted(np.unique(dates[labels == g]).tolist()) for g in range(4)}
configs = []
for r in range(2):   # two rotations: hold out the last, then the first, date of every grade
    configs.append({g: (gdates[g][-1] if r == 0 else gdates[g][0]) for g in range(4) if len(gdates[g]) >= 2})
for ci, cfg in enumerate(configs):
    if len(cfg) < 4: print("skip config, some grade has a single date", cfg); continue
    te = np.array([i for i in range(N) if dates[i] == cfg[labels[i]]])
    rest = np.array([i for i in range(N) if dates[i] != cfg[labels[i]] and dates[i] not in {cfg[g] for g in cfg}])
    if len(rest) == 0 or len(np.unique(labels[rest])) < 4:
        rest = np.array([i for i in range(N) if dates[i] != cfg[labels[i]]])   # fallback: only same-grade date held out
        mode_note = "held-out date removed per grade only"
    else:
        mode_note = "held-out dates removed for all grades"
    a, b = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED).split(rest, labels[rest], groups[rest]))
    for mid, mname, seeds in [(R18_ID, "ResNet-18", SEEDS), (EB0_ID, "EfficientNet-B0", SEEDS[:1])]:
        for sd in seeds:
            key = f"cfg{ci}|{mname}|{sd}"
            if key in C: continue
            X_t = X_orig
            res, prob = train_eval(mid, rest[a], rest[b], te, "rgb", seed=sd)
            C[key] = {"held_out": {GRADES[g]: cfg[g] for g in cfg}, "note": mode_note, "n_train": int(len(rest)),
                      **summarize(labels[te], prob.argmax(1)), "val_macro_f1": res["val_macro_f1"]}
            save("11_C_leavedate", C); print("leave-date", key, round(C[key]["macro_f1"] * 100, 2), flush=True)

# %% [markdown]
# ### D. Silhouette controls on the fixed grouped split

# %%
D = load("11_D_silhouette") if done("11_D_silhouette") else {}
for variant, T in [("background_inpainted", INP_t), ("silhouette_only", SIL_t)]:
    for sd in SEEDS:
        key = f"{variant}|ResNet-18|{sd}"
        if key in D: continue
        X_t = T; res, prob = train_eval(R18_ID, tr_idx, va_idx, te_idx, "rgb", seed=sd)
        D[key] = summarize(labels[te_idx], prob.argmax(1)); save("11_D_silhouette", D)
        print("silhouette", key, round(D[key]["macro_f1"] * 100, 2), flush=True)
X_t = X_orig
if "shape_features" not in D:
    sf = []
    for i in range(N):
        lab_ = sklabel(fgm[i]); props = regionprops(lab_)
        if not props: sf.append([0] * 6); continue
        r = max(props, key=lambda q: q.area)
        sf.append([r.area, r.perimeter, r.eccentricity, r.major_axis_length, r.minor_axis_length, r.solidity])
    sf = np.array(sf, float); cvp = np.zeros(N, int)
    for trv, te in StratifiedGroupKFold(n_splits=NF, shuffle=True, random_state=SEED).split(paths, labels, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(sf[trv], labels[trv])
        cvp[te] = clf.predict(sf[te])
    D["shape_features"] = {"features": "area, perimeter, eccentricity, major and minor axis, solidity", **summarize(labels, cvp)}
    save("11_D_silhouette", D); print("shape features macro-F1", round(D["shape_features"]["macro_f1"] * 100, 2))

# %% [markdown]
# ### E. Swap test with paste-back control, repeated random pairings, binomial test

# %%
E = load("11_E_swap") if done("11_E_swap") else {}
def composite(i, j):
    a = INP[j].copy(); d = tight[i]; a[d] = X[i][d]; return a
te_lab = labels[te_idx]
for mid, mname in [(R18_ID, "ResNet-18"), (EB0_ID, "EfficientNet-B0")]:
    for sd in SEEDS:
        key = f"{mname}|{sd}"
        if key in E: continue
        X_t = X_orig
        res, prob, mdl = train_eval(mid, tr_idx, va_idx, te_idx, "rgb", seed=sd, return_model=True)
        def pred_on(arrs):
            global X_t
            T = X_orig.clone()
            for i, a_ in zip(te_idx, arrs): T[i] = torch.from_numpy(a_).permute(2, 0, 1)
            keep = X_t; X_t = T
            try: return predict(mdl, te_idx, "rgb").argmax(1)
            finally: X_t = keep
        pb = pred_on([composite(i, i) for i in te_idx])                       # paste-back control
        out = {"original_acc": float((prob.argmax(1) == te_lab).mean()), "pasteback": summarize(te_lab, pb), "pairings": []}
        rng = np.random.default_rng(sd)
        for rep in range(1 if TEST_MODE else 5):
            partner = np.array([int(rng.choice(te_idx[te_lab != labels[i]])) for i in te_idx])
            ps = pred_on([composite(i, j) for i, j in zip(te_idx, partner)])
            bgl = labels[partner]; nb, nc = int((ps == bgl).sum()), int((ps == te_lab).sum())
            out["pairings"].append({"follows_background": nb / len(ps), "follows_clove": nc / len(ps),
                                    "neither": 1 - (nb + nc) / len(ps),
                                    "binomial_p_bg_vs_clove": float(_binom(nb, nb + nc, 0.5).pvalue) if nb + nc else 1.0})
        fb = [q["follows_background"] for q in out["pairings"]]; fc = [q["follows_clove"] for q in out["pairings"]]
        out["mean_follows_background"], out["mean_follows_clove"] = float(np.mean(fb)), float(np.mean(fc))
        E[key] = out; save("11_E_swap", E)
        print("swap", key, "paste-back acc", round(out["pasteback"]["accuracy"] * 100, 1), "bg", round(out["mean_follows_background"] * 100, 1),
              "clove", round(out["mean_follows_clove"] * 100, 1), flush=True)
        del mdl
        if dev.type == "cuda": torch.cuda.empty_cache()
X_t = X_orig

zp = os.path.join(OUT, "..", "results_session.zip")
with zipfile.ZipFile(zp, "w") as z:
    for f in glob.glob(os.path.join(OUT, "11_*.json")): z.write(f, os.path.basename(f))
print("DONE. Download", os.path.abspath(zp))
