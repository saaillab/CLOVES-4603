# %% [markdown]
# # Zanzibar clove grading: session-confound check (run after clove_validation) for the Smart Agricultural Technology resubmission
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
OUT       = os.environ.get("CLOVE_OUT", "/kaggle/working/results_confound")
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
# ## 9. Session-confound check
# Each grade was photographed in its own session. If the background alone predicts the grade, part of the
# benchmark accuracy reflects session conditions, not clove quality.
# - **original**: unchanged images
# - **background_only**: clove (dilated mask) painted over with that image's median background colour
# - **clove_only**: every background pixel set to the same constant paper colour
# Same grouped split and protocol as the benchmark, plus 5-fold grouped CV with ResNet-18.

# %%
X_orig = X_t.clone()
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
dil = np.stack([cv2.dilate((M_t[i, 0].numpy() > 0.5).astype(np.uint8), k) for i in range(N)]).astype(bool)
bg_only = X.copy(); cl_only = X.copy()
for i in range(N):
    d = dil[i]
    if d.all() or not d.any(): continue
    med = np.median(X[i][~d], axis=0).astype(np.uint8)
    bg_only[i][d] = med
    cl_only[i][~d] = np.array([235, 235, 235], np.uint8)
VARIANTS = {"original": X_orig,
            "background_only": torch.from_numpy(bg_only).permute(0, 3, 1, 2).contiguous(),
            "clove_only": torch.from_numpy(cl_only).permute(0, 3, 1, 2).contiguous()}
save("09_mask_coverage", {"dilated_fraction_mean": float(dil.mean()), "images_skipped": int(sum(dil[i].all() or not dil[i].any() for i in range(N)))})

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ex = [np.where(labels == g)[0][0] for g in range(4)]
fig, ax = plt.subplots(3, 4, figsize=(10, 7.5))
for r, (nm, V) in enumerate(VARIANTS.items()):
    for c, i in enumerate(ex):
        ax[r, c].imshow(V[i].permute(1, 2, 0).numpy()); ax[r, c].axis("off")
        ax[r, c].set_title(f"{nm} | {GRADES[labels[i]]}", fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "confound_examples.png"), dpi=150); plt.close()

MODELS = [("ResNet-18", "resnet18.tv_in1k" if PRETRAINED else "resnet18"),
          ("EfficientNet-B0", "efficientnet_b0.ra_in1k" if PRETRAINED else "efficientnet_b0")]
for vn, V in VARIANTS.items():
    X_t = V
    for disp, tid in MODELS:
        key = f"09_split_{vn}_{disp}"
        if done(key): continue
        res, prob = train_eval(tid, tr_idx, va_idx, te_idx, "rgb")
        save(key, {**res, **metrics(labels[te_idx], prob.argmax(1))})
        print(f"{vn:16s} {disp:16s} test macro-F1 {load(key)['macro_f1']*100:.2f}", flush=True)

NF = 2 if TEST_MODE else 5
folds = list(StratifiedGroupKFold(n_splits=NF, shuffle=True, random_state=SEED).split(paths, labels, groups))
cvc = load("09_cv") if done("09_cv") else {}
for vn, V in VARIANTS.items():
    X_t = V; cvc.setdefault(vn, [])
    for f, (trv, te) in enumerate(folds):
        if f < len(cvc[vn]): continue
        a, b = next(StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED + f).split(paths[trv], labels[trv], groups[trv]))
        res, prob = train_eval(MODELS[0][1], trv[a], trv[b], te, "rgb", seed=SEED + f)
        cvc[vn].append(float(f1_score(labels[te], prob.argmax(1), average="macro"))); save("09_cv", cvc)
        print(f"CV {vn} fold {f+1}: {cvc[vn][-1]*100:.2f}", flush=True)
X_t = X_orig

summary = {vn: {"cv_mean": float(np.mean(v)), "cv_sd": float(np.std(v, ddof=1)), "folds": v} for vn, v in cvc.items()}
summary["chance_macro_f1"] = 0.25
summary["split"] = {k[len("09_split_"):]: load(k)["macro_f1"] for k in [f"09_split_{vn}_{d}" for vn in VARIANTS for d, _ in MODELS]}
save("09_summary", summary); print(json.dumps(summary, indent=2))
with zipfile.ZipFile("/kaggle/working/results_confound.zip" if not TEST_MODE else os.path.join(OUT, "..", "results_confound.zip"), "w") as z:
    for f in glob.glob(os.path.join(OUT, "*")):
        if not f.endswith(".pt"): z.write(f, os.path.basename(f))
print("DONE. Download results_confound.zip from the Output tab.")
