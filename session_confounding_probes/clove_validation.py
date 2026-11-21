# %% [markdown]
# # Zanzibar clove grading: leak-free validation for the Smart Agricultural Technology resubmission
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
OUT       = os.environ.get("CLOVE_OUT", "/kaggle/working/results")
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
# ## 2. 18-architecture benchmark (identical protocol; only the architecture changes)

# %%
BENCH = [  # (display name, timm id)
    ("ResNet-18", "resnet18.tv_in1k"), ("ResNet-34", "resnet34.tv_in1k"),
    ("ResNet-50", "resnet50.tv_in1k"), ("ResNet-101", "resnet101.tv_in1k"),
    ("DenseNet-121", "densenet121.tv_in1k"), ("DenseNet-169", "densenet169.tv_in1k"), ("DenseNet-201", "densenet201.tv_in1k"),
    ("EfficientNet-B0", "efficientnet_b0.ra_in1k"), ("EfficientNet-B1", "efficientnet_b1.ft_in1k"),
    ("EfficientNet-B2", "efficientnet_b2.ra_in1k"), ("EfficientNet-B3", "efficientnet_b3.ra2_in1k"),
    ("Xception", "legacy_xception.tf_in1k"),
    ("ViT-Base/16", "vit_base_patch16_224.augreg2_in21k_ft_in1k"), ("DeiT-Base", "deit_base_patch16_224.fb_in1k"),
    ("Swin-Base", "swin_base_patch4_window7_224.ms_in22k_ft_in1k"),
    ("VGG-16", "vgg16.tv_in1k"), ("VGG-16-BN", "vgg16_bn.tv_in1k"),
]
if TEST_MODE: BENCH = [("ResNet-18", "resnet18"), ("EfficientNet-B0", "efficientnet_b0")]
# Preflight (~3 min): download every model's weights and run one train step on GPU, so problems show up now, not hours in.
if not TEST_MODE and not done("02_preflight"):
    bad = {}
    for disp, tid in BENCH:
        try:
            m = make_model(tid); opt = torch.optim.AdamW(m.parameters(), lr=LR)
            xb = X_t[tr_idx[:BS]].to(dev); yb = Y_t[tr_idx[:BS]].to(dev)
            with torch.autocast(dev.type, enabled=AMP): loss = F.cross_entropy(m(augment(to_input(xb))), yb)
            loss.backward(); opt.step(); del m, opt; torch.cuda.empty_cache()
        except Exception as e:
            bad[disp] = repr(e)
    for mode, c in [("masked", 3), ("rgbm", 4)]:
        m = make_model(R18_ID, c); xb = X_t[tr_idx[:BS]].to(dev); mb = torch.ones(len(xb), 1, IMG, IMG, device=dev)
        F.cross_entropy(m(augment(to_input(xb, mb, mode))), Y_t[tr_idx[:BS]].to(dev)).backward(); del m
    if bad: raise SystemExit("Preflight failed, nothing trained yet: " + json.dumps(bad, indent=1))
    save("02_preflight", {"ok": [d for d, _ in BENCH]}); print("Preflight OK for all", len(BENCH), "models")
    if os.environ.get("CLOVE_PREFLIGHT_ONLY") == "1": raise SystemExit("preflight-only run finished")
bench_probs = {}
for disp, tid in BENCH:
    key = "02_bench_" + disp.replace("/", "_")
    if done(key):
        bench_probs[disp] = np.array(load(key)["test_probs"]); print("skip", disp); continue
    print(f"--- {disp} ({tid})", flush=True)
    try:
        res, prob = train_eval(tid, tr_idx, va_idx, te_idx, "rgb")
    except Exception as e:                                   # record and continue; never silently drop a model
        print("FAILED", disp, repr(e)); save(key + "_FAILED", {"error": repr(e)}); continue
    res.update({"model": disp, "timm_id": tid, **metrics(labels[te_idx], prob.argmax(1)), "test_probs": prob.tolist()})
    save(key, res); bench_probs[disp] = prob
    print(f"{disp}: test macro-F1 {res['macro_f1']*100:.2f} CI {np.round(np.array(res['macro_f1_ci95'])*100,2)} "
          f"best epoch {res['best_epoch']} ({res['train_minutes']:.1f} min)", flush=True)
    if dev.type == "cuda": torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Classical baseline: RBF-SVM on LAB statistics + GLCM (class-balanced; tuned on validation only)

# %%
if not done("03_svm"):
    from skimage.color import rgb2lab
    from skimage.feature import graycomatrix, graycoprops
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    def feats(a):
        lab = rgb2lab(a).reshape(-1, 3)
        f = list(lab.mean(0)) + list(lab.std(0)) + list(np.percentile(lab, [10, 50, 90], axis=0).ravel())
        g = (np.asarray(Image.fromarray(a).convert("L")) // 8).astype(np.uint8)
        P = graycomatrix(g, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
        for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]:
            f += list(graycoprops(P, prop).ravel())
        return f
    Fm = np.array([feats(X[k]) for k in range(N)], np.float32)
    sc = StandardScaler().fit(Fm[tr_idx])
    best = (-1, None)
    for C in [0.1, 1, 10, 100]:
        for gm in ["scale", 0.01, 0.001]:
            clf = SVC(C=C, gamma=gm, kernel="rbf", class_weight="balanced").fit(sc.transform(Fm[tr_idx]), labels[tr_idx])
            v = f1_score(labels[va_idx], clf.predict(sc.transform(Fm[va_idx])), average="macro")
            if v > best[0]: best = (v, (C, gm))
    clf = SVC(C=best[1][0], gamma=best[1][1], kernel="rbf", class_weight="balanced").fit(sc.transform(Fm[tr_idx]), labels[tr_idx])
    pred = clf.predict(sc.transform(Fm[te_idx]))
    save("03_svm", {"model": "SVM (LAB+GLCM)", "best_C_gamma": best[1], "val_macro_f1": best[0], "n_features": Fm.shape[1],
                    **metrics(labels[te_idx], pred), "test_pred": pred.tolist()})
    print("SVM test macro-F1", load("03_svm")["macro_f1"])

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
# ## 4. 10-fold grouped CV, all three arms retrained inside every fold on identical splits
# - **monolithic**: RGB ResNet-18
# - **masked**: RGB with background zeroed (does the gain come only from removing background?)
# - **decomposed**: RGB + mask channel (the paper's Stage 2)

# %%
ARMS = [("monolithic", "rgb"), ("masked", "masked"), ("decomposed", "rgbm")]
R18 = R18_ID
cv = load("04_cv") if done("04_cv") else {"folds": []}
skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(paths, labels, groups))
cam_models = {}
for k, (trv, te) in enumerate(folds):
    if k < len(cv["folds"]): continue
    inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED + k)
    a, b = next(inner.split(paths[trv], labels[trv], groups[trv])); tr, va = trv[a], trv[b]
    rec = {"fold": k + 1, "n_test": int(len(te)), "test_idx": te.tolist()}
    for arm, mode in ARMS:
        res, prob, mdl = train_eval(R18, tr, va, te, mode, M_t, seed=SEED + k, return_model=True)
        pred = prob.argmax(1)
        rec[arm] = {**res, "macro_f1": float(f1_score(labels[te], pred, average="macro")), "pred": pred.tolist()}
        if k == 0 and arm in ("monolithic", "decomposed"):
            torch.save(mdl.state_dict(), os.path.join(OUT, f"fold1_{arm}.pt"))
    cv["folds"].append(rec); save("04_cv", cv)
    print(f"Fold {k+1}: " + " | ".join(f"{a} {rec[a]['macro_f1']*100:.2f}" for a, _ in ARMS), flush=True)

def mean_ci(v):
    v = np.array(v); h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci95": [float(v.mean() - h), float(v.mean() + h)]}
summ = {a: mean_ci([f[a]["macro_f1"] for f in cv["folds"]]) for a, _ in ARMS}
yt = np.concatenate([labels[np.array(f["test_idx"])] for f in cv["folds"]])
P = {a: np.concatenate([f[a]["pred"] for f in cv["folds"]]) for a, _ in ARMS}
tests = {}
for a, b in [("decomposed", "monolithic"), ("decomposed", "masked"), ("masked", "monolithic")]:
    fa = [f[a]["macro_f1"] for f in cv["folds"]]; fb = [f[b]["macro_f1"] for f in cv["folds"]]
    tt = ttest_rel(fa, fb)
    try: wx = wilcoxon(fa, fb).pvalue
    except ValueError: wx = 1.0                               # all differences zero
    tests[f"{a}_vs_{b}"] = {"mean_diff_pp": float(np.mean(np.array(fa) - np.array(fb)) * 100),
                            "paired_t": float(tt.statistic), "paired_t_p": float(tt.pvalue), "wilcoxon_p": float(wx),
                            "mcnemar_pooled_oof": mcnemar_exact(yt, P[a], P[b])}
cv_summary = {"per_arm": summ, "tests": tests, "pooled_oof": {a: metrics(yt, P[a]) for a, _ in ARMS},
              "note": "Paired tests across CV folds are anti-conservative (overlapping training sets); report with the CI and McNemar."}
save("04_cv_summary", cv_summary); print(json.dumps({"per_arm": summ, "tests": tests}, indent=2))

# %% [markdown]
# ## 5. Grad-CAM (fold-1 models): share of CAM energy outside the U-Net clove mask

# %%
def gradcam(model, x, layer):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    out = model(x); model.zero_grad(); out.gather(1, out.argmax(1, keepdim=True)).sum().backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True); cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    return F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

if not done("05_gradcam") and os.path.exists(os.path.join(OUT, "fold1_monolithic.pt")):
    te = np.array(cv["folds"][0]["test_idx"]); out = {}
    for arm, mode in [("monolithic", "rgb"), ("decomposed", "rgbm")]:
        m = make_model(R18, 4 if mode == "rgbm" else 3); m.load_state_dict(torch.load(os.path.join(OUT, f"fold1_{arm}.pt"), map_location=dev)); m.eval()
        frac = []
        for s in range(0, len(te), 16):
            b = te[s:s + 16]; xb = X_t[b].to(dev); mb = M_t[b].to(dev)
            cam = gradcam(m, to_input(xb, mb, mode).requires_grad_(True), m.layer4).detach()
            inside = (cam * (mb > 0.5)).sum((1, 2, 3)); total = cam.sum((1, 2, 3)) + 1e-8
            frac += (1 - inside / total).cpu().tolist()
        out[arm] = {"outside_mask_energy_mean": float(np.mean(frac)), "sd": float(np.std(frac, ddof=1)), "n": len(frac)}
        if arm == "monolithic":   # save a few example overlays for the paper figure
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            b = te[:4]; xb = X_t[b].to(dev); mb = M_t[b].to(dev)
            cam = gradcam(m, to_input(xb, mb, mode).requires_grad_(True), m.layer4).detach().cpu()
            fig, ax = plt.subplots(2, 4, figsize=(10, 5))
            for j in range(len(b)):
                ax[0, j].imshow(X[b[j]]); ax[0, j].set_title(GRADES[labels[b[j]]]); ax[1, j].imshow(X[b[j]])
                ax[1, j].imshow(cam[j, 0], alpha=0.5, cmap="jet"); ax[0, j].axis("off"); ax[1, j].axis("off")
            plt.tight_layout(); plt.savefig(os.path.join(OUT, "gradcam_monolithic_examples.png"), dpi=200); plt.close()
    save("05_gradcam", out); print(out)

# %% [markdown]
# ## 6. CPU latency, batch size 1, one thread (edge-device proxy; a Jetson/Raspberry Pi measurement is still better)

# %%
if not done("06_latency"):
    torch.set_num_threads(1); lat = {}
    cands = [("EfficientNet-B0", "efficientnet_b0", 3), ("ResNet-18", "resnet18", 3), ("ResNet-50", "resnet50", 3),
             ("DenseNet-201", "densenet201", 3), ("Decomposed classifier (ResNet-18, 4ch)", "resnet18", 4)]
    import timm
    x3, x4 = torch.randn(1, 3, 224, 224), torch.randn(1, 4, 224, 224)
    for disp, tid, c in cands:
        m = timm.create_model(tid, pretrained=False, num_classes=4, in_chans=c).eval(); x = x4 if c == 4 else x3
        with torch.no_grad():
            for _ in range(5): m(x)
            ts = []
            for _ in range(10 if TEST_MODE else 50): t0 = time.perf_counter(); m(x); ts.append((time.perf_counter() - t0) * 1000)
        lat[disp] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}
    u = EnhancedUNet().eval()
    with torch.no_grad():
        ts = []
        for _ in range(5 if TEST_MODE else 20): t0 = time.perf_counter(); u(x3); ts.append((time.perf_counter() - t0) * 1000)
    lat["U-Net (Stage 1)"] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}
    lat["cpu"] = os.popen("lscpu | grep 'Model name'").read().strip() or "unknown"
    torch.set_num_threads(os.cpu_count()); save("06_latency", lat); print(json.dumps(lat, indent=2))

# %% [markdown]
# ## 7. Batch-level ZSTC grading (only runs if BATCH_DIR exists with Grade 1..Grade 4 subfolders)
# Stage 1 U-Net -> connected components -> each clove crop graded by a decomposed classifier trained on 8/9 of
# the single-clove images (batch photos are never used in training) -> ZSTC mpeta-density rule.


# %%
if not done("07_batch") and os.path.isdir(BATCH_DIR):
    import cv2
    bp, bl = [], []
    for i, g in enumerate(BATCHES):
        d = os.path.join(BATCH_DIR, g)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")): bp.append(os.path.join(d, f)); bl.append(i)
    bl = np.array(bl); print("batch images per grade", np.bincount(bl, minlength=4).tolist())
    allidx = np.arange(N); g2 = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    a, b = next(g2.split(paths, labels, groups))
    _, _, clf = train_eval(R18, a, b, b, "rgbm", M_t, return_model=True)
    def rule(rho):  # ZSTC thresholds on % of mpeta (Grade IV) cloves
        return 0 if rho <= 3 else 1 if rho <= 7 else 2 if rho <= 20 else 3
    rows = []
    for p, y in zip(bp, bl):
        im = np.asarray(Image.open(p).convert("RGB")); Hh, Ww = im.shape[:2]
        S = 512; small = np.asarray(Image.fromarray(im).resize((S, S)))
        mk = segment(small) > 0.5
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), 8)
        area = st[1:, cv2.CC_STAT_AREA]; keep = [j + 1 for j in range(n - 1) if area[j] >= max(20, 0.2 * np.median(area))] if n > 1 else []
        crops, cm = [], []
        for j in keep:
            x0, y0, w, h = st[j, :4]; sx, sy = Ww / S, Hh / S; pad = 4
            X0, Y0 = max(0, int((x0 - pad) * sx)), max(0, int((y0 - pad) * sy)); X1, Y1 = min(Ww, int((x0 + w + pad) * sx)), min(Hh, int((y0 + h + pad) * sy))
            crops.append(np.asarray(Image.fromarray(im[Y0:Y1, X0:X1]).resize((IMG, IMG))))
            mm = (lab[max(0, y0 - pad):y0 + h + pad, max(0, x0 - pad):x0 + w + pad] == j).astype(np.float32)
            cm.append(np.asarray(Image.fromarray(mm).resize((IMG, IMG))))
        if not crops: rows.append({"file": os.path.basename(p), "true": int(y), "n_cloves": 0, "pred": -1}); continue
        xb = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(dev); mb = torch.from_numpy(np.stack(cm))[:, None].to(dev)
        with torch.no_grad(): pc = clf(to_input(xb, mb, "rgbm")).argmax(1).cpu().numpy()
        rho = 100 * (pc == 3).mean()
        rows.append({"file": os.path.basename(p), "true": int(y), "n_cloves": len(pc), "mpeta_pct": float(rho),
                     "per_clove_counts": np.bincount(pc, minlength=4).tolist(), "pred": rule(rho)})
    yb = np.array([r["true"] for r in rows]); pb = np.array([r["pred"] for r in rows])
    save("07_batch", {"n": len(rows), "per_grade": np.bincount(yb, minlength=4).tolist(),
                      "rule_based": metrics(yb, np.where(pb < 0, 0, pb)), "no_clove_found": int((pb < 0).sum()),
                      "median_cloves_per_image": float(np.median([r["n_cloves"] for r in rows])), "rows": rows})
    print("batch rule-based macro-F1", load("07_batch")["rule_based"]["macro_f1"])
elif not os.path.isdir(BATCH_DIR):
    print("BATCH_DIR not found, section 7 skipped:", BATCH_DIR)

# %% [markdown]
# ## 8. Summary tables and results.zip (download and send this file plus the notebook log)

# %%
rows = []
for disp, _ in BENCH:
    k = "02_bench_" + disp.replace("/", "_")
    if done(k):
        r = load(k); rows.append((disp, r["params_M"], r["macro_f1"], r["macro_f1_ci95"], r["accuracy"], r["best_epoch"]))
if rows:
    best = max(rows, key=lambda r: r[2])[0]; yte = labels[te_idx]
    pv = {d: mcnemar_exact(yte, bench_probs[best].argmax(1), bench_probs[d].argmax(1))["p"] for d, *_ in rows if d != best}
    adj = dict(zip(pv, holm(list(pv.values())))) if pv else {}
    lines = ["| Model | Params (M) | Macro-F1 % [95% CI] | Acc % | Best epoch | McNemar vs best (Holm p) |", "|---|---|---|---|---|---|"]
    for d, pm, f, ci, acc, be in sorted(rows, key=lambda r: -r[2]):
        lines.append(f"| {d} | {pm:.1f} | {f*100:.2f} [{ci[0]*100:.2f}, {ci[1]*100:.2f}] | {acc*100:.2f} | {be} | "
                     + ("best" if d == best else f"{adj[d]:.3g}") + " |")
    if done("03_svm"):
        s = load("03_svm"); lines.append(f"| SVM (LAB+GLCM) | n/a | {s['macro_f1']*100:.2f} [{s['macro_f1_ci95'][0]*100:.2f}, {s['macro_f1_ci95'][1]*100:.2f}] | {s['accuracy']*100:.2f} | n/a | n/a |")
    table = "\n".join(lines); print(table); open(os.path.join(OUT, "benchmark_table.md"), "w").write(table)
if done("04_cv_summary"):
    s = load("04_cv_summary")
    for a in s["per_arm"]:
        v = s["per_arm"][a]; print(f"CV {a}: {v['mean']*100:.2f} +/- {v['sd']*100:.2f} (95% CI {v['ci95'][0]*100:.2f} to {v['ci95'][1]*100:.2f})")
    for t, v in s["tests"].items(): print(t, {k: v[k] for k in ("mean_diff_pp", "paired_t_p", "wilcoxon_p")}, v["mcnemar_pooled_oof"])
with zipfile.ZipFile("/kaggle/working/results.zip" if not TEST_MODE else os.path.join(OUT, "..", "results.zip"), "w") as z:
    for f in glob.glob(os.path.join(OUT, "*")):
        if not f.endswith(".pt"): z.write(f, os.path.basename(f))
print("DONE. Download results.zip from the Output tab.")
