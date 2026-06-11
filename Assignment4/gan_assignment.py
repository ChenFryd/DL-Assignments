# %% [markdown]
# # Assignment 4 – Generative Models with GAN
# **Ben-Gurion University, Deep Learning Course**
#
# Implements a standard GAN and Conditional GAN (cGAN) on the Adult Census Income
# tabular dataset, then investigates discrete-feature representation strategies,
# mode-collapse dynamics, and a spectral-normalization variant.

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# %% [markdown]
# ## 2. Data Loading & Exploration

# %%
def load_arff(path: str) -> pd.DataFrame:
    """Parse a minimal ARFF file into a DataFrame.
    Skips all @ header lines, uses @Attribute names as columns,
    and treats '?' as NaN.
    """
    from io import StringIO
    col_names, in_data, rows = [], False, []
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('%'):
                continue
            if s.upper().startswith('@ATTRIBUTE'):
                col_names.append(s.split()[1])
            elif s.upper().startswith('@DATA'):
                in_data = True
            elif in_data:
                rows.append(s)
    return pd.read_csv(StringIO('\n'.join(rows)), header=None,
                       names=col_names, na_values=['?'], skipinitialspace=True)


df_raw = load_arff('adult.arff')
print(f"Raw shape: {df_raw.shape}")
print("\nClass distribution:")
print(df_raw['income'].value_counts())

# %%
df_raw.head()

# %%
df_raw.describe()

# %% [markdown]
# ## 2b. Exploratory Data Analysis

# %%
# Class distribution
fig, ax = plt.subplots(figsize=(5, 4))
income_clean = df_raw['income'].astype(str).str.strip().str.rstrip('.')
counts = income_clean.value_counts()
bars = ax.bar(counts.index, counts.values, color=['steelblue', 'coral'], alpha=0.85)
for bar, val in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 0.5,
        f'{val}\n({val/len(df_raw)*100:.1f}%)',
        ha='center', va='center', fontsize=10, color='white', fontweight='bold'
    )
ax.set_title('Class Distribution – income')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_class_dist.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

# %%
# Continuous feature distributions
_cont_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for i, col in enumerate(_cont_cols):
    ax = axes[i // 3][i % 3]
    ax.hist(df_raw[col].dropna(), bins=40, color='steelblue', alpha=0.8, density=True)
    ax.set_title(col); ax.grid(True, alpha=0.3)
plt.suptitle('Real Data – Continuous Feature Distributions')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_continuous.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

# %%
# Categorical feature distributions (top 15 per feature; remainder grouped as Other)
_cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'native-country']
TOP_N = 15
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
for i, col in enumerate(_cat_cols):
    ax = axes[i // 4][i % 4]
    vc = df_raw[col].astype(str).str.strip().value_counts()
    if len(vc) > TOP_N:
        top = vc.iloc[:TOP_N]
        other_count = vc.iloc[TOP_N:].sum()
        import pandas as pd
        vc = pd.concat([top, pd.Series({'Other': other_count})])
    ax.bar(range(len(vc)), vc.values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=45, ha='right', fontsize=6)
    ax.set_title(col, fontsize=9); ax.grid(True, alpha=0.3, axis='y')
plt.suptitle('Real Data – Categorical Feature Distributions (top 15 shown)')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_categorical.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

print(f"\nDataset summary:")
print(f"  Total records  : {len(df_raw)}")
print(f"  Continuous cols: {len(_cont_cols)}")
print(f"  Categorical cols: {len(_cat_cols)}")
print(f"  Missing values :\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0]}")

# %% [markdown]
# ## 3. Preprocessing
#
# **Design decisions:**
# - **Continuous features** (age, fnlwgt, education-num, capital-gain, capital-loss,
#   hours-per-week): `StandardScaler` → roughly N(0,1). The generator uses linear
#   activations for these heads so it can produce any real value.
# - **Categorical features** (8 columns): one-hot encoding using the full vocabulary
#   built from the complete dataset to ensure consistent dimensions across splits.
# - Missing / unknown categorical values are mapped to an "Unknown" category.

# %%
CONTINUOUS_COLS = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                   'capital-loss', 'hours-per-week']
CATEGORICAL_COLS = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country']
TARGET_COL = 'income'
NUM_CLASSES = 2


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = (
        df[TARGET_COL].astype(str).str.strip().str.rstrip('.')
        .map({'<=50K': 0, '>50K': 1})
    )
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).replace('nan', 'Unknown').fillna('Unknown')
    return df.dropna(subset=[TARGET_COL]).reset_index(drop=True)


df = prepare_df(df_raw)
print(f"Clean shape: {df.shape}")
print(df[TARGET_COL].value_counts(normalize=True).round(3))

# %%
# Build vocabularies from the full dataset for consistent one-hot dimensions
CAT_VOCABS  = {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_COLS}
CAT_DIMS    = {col: len(v) for col, v in CAT_VOCABS.items()}
CAT_OFFSETS: dict[str, int] = {}
_off = 0
for col in CATEGORICAL_COLS:
    CAT_OFFSETS[col] = _off
    _off += CAT_DIMS[col]

CONT_DIM = len(CONTINUOUS_COLS)   # 6
CAT_DIM  = sum(CAT_DIMS.values())
DATA_DIM = CONT_DIM + CAT_DIM

print(f"Continuous dim : {CONT_DIM}")
print(f"Categorical dim: {CAT_DIM}  (breakdown: {dict(list(CAT_DIMS.items())[:4])} ...)")
print(f"Total data dim : {DATA_DIM}")

# %%
def encode_df(df_sub: pd.DataFrame, scaler=None):
    """
    Encode to float32 array of shape [N, DATA_DIM].
    Layout: [StandardScaler(continuous) | one-hot(categorical)]
    Returns (X, y, scaler).
    """
    X_cont = df_sub[CONTINUOUS_COLS].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        X_cont = scaler.fit_transform(X_cont)
    else:
        X_cont = scaler.transform(X_cont)

    N = len(df_sub)
    X_cat = np.zeros((N, CAT_DIM), dtype=np.float32)
    for col in CATEGORICAL_COLS:
        vocab   = CAT_VOCABS[col]
        idx_map = {v: i for i, v in enumerate(vocab)}
        for i, val in enumerate(df_sub[col].astype(str).values):
            X_cat[i, CAT_OFFSETS[col] + idx_map.get(val, 0)] = 1.0

    X = np.concatenate([X_cont, X_cat], axis=1)
    y = df_sub[TARGET_COL].values.astype(np.float32)
    return X, y, scaler


def decode_to_df(X: np.ndarray, scaler: StandardScaler) -> pd.DataFrame:
    """Decode encoded array back to a human-readable DataFrame."""
    X_cont_dec = scaler.inverse_transform(X[:, :CONT_DIM])
    df_out = pd.DataFrame(X_cont_dec, columns=CONTINUOUS_COLS)
    for col in CATEGORICAL_COLS:
        start = CONT_DIM + CAT_OFFSETS[col]
        end   = start + CAT_DIMS[col]
        idxs  = np.argmax(X[:, start:end], axis=1)
        df_out[col] = [CAT_VOCABS[col][i] for i in idxs]
    return df_out

# %% [markdown]
# ## 4. Model Architecture
#
# ### Design rationale
# - **Generator body**: 3-layer MLP with BatchNorm + LeakyReLU.
# - **Continuous head**: linear (no activation) so the generator can produce
#   unbounded values matching the StandardScaler output range.
# - **Categorical heads**: one separate linear layer per categorical feature,
#   followed by either *Softmax* or *Gumbel-Softmax with straight-through estimator*
#   (the default). Gumbel-Softmax is preferred because it produces hard one-hot
#   samples in the forward pass (matching the real data format seen by the
#   discriminator) while keeping differentiable gradients in the backward pass.
# - **Discriminator**: 3-layer MLP with LeakyReLU + Dropout (no BatchNorm — BN
#   interacts poorly with the discriminator's real/fake predictions per sample).
# - **cGAN extension**: the income one-hot (dim=2) is concatenated to the noise
#   fed to the generator and to the data fed to the discriminator.

# %%
def gumbel_softmax_st(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Gumbel-Softmax with straight-through estimator.
    Forward  : hard one-hot (argmax of Gumbel-perturbed logits).
    Backward : gradients flow through the soft Gumbel-Softmax.
    """
    U = torch.rand_like(logits).clamp(1e-20, 1.0 - 1e-20)
    gumbel_noise = -torch.log(-torch.log(U))
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    y_hard = torch.zeros_like(y_soft).scatter_(
        -1, y_soft.argmax(dim=-1, keepdim=True), 1.0
    )
    # Straight-through: hard forward, soft backward
    return y_hard - y_soft.detach() + y_soft


class Generator(nn.Module):
    """
    Mixed-output generator for tabular data.
    Continuous features : linear head (unbounded).
    Categorical features: separate head per feature → Gumbel-ST or Softmax.
    For cGAN pass label_dim > 0; the income one-hot is concatenated to noise.
    """
    def __init__(self, z_dim: int = 128, label_dim: int = 0,
                 hidden: int = 256, cat_strategy: str = 'gumbel',
                 temperature: float = 1.0):
        super().__init__()
        self.z_dim        = z_dim
        self.cat_strategy = cat_strategy
        self.temperature  = temperature

        inp = z_dim + label_dim
        self.body = nn.Sequential(
            nn.Linear(inp, hidden),    nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2, inplace=True),
        )
        self.cont_head = nn.Linear(hidden, CONT_DIM)   # linear, no activation
        self.cat_heads = nn.ModuleList(
            [nn.Linear(hidden, CAT_DIMS[col]) for col in CATEGORICAL_COLS]
        )

    def forward(self, z: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        inp = torch.cat([z, label], dim=1) if label is not None else z
        h   = self.body(inp)
        cont = self.cont_head(h)
        cats = []
        for head in self.cat_heads:
            logits = head(h)
            if self.cat_strategy == 'gumbel':
                cats.append(gumbel_softmax_st(logits, self.temperature))
            else:   # 'softmax'
                cats.append(F.softmax(logits, dim=-1))
        return torch.cat([cont] + cats, dim=1)


class Discriminator(nn.Module):
    """Standard binary discriminator (real=1 / fake=0)."""
    def __init__(self, data_dim: int = DATA_DIM, label_dim: int = 0, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + label_dim, hidden), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, hidden),               nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),          nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, 1),               nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        inp = torch.cat([x, label], dim=1) if label is not None else x
        return self.net(inp).squeeze(1)

# %% [markdown]
# ## 5. Training Loop

# %%
def train_gan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int     = 200,
    batch_size: int   = 256,
    z_dim: int        = 128,
    lr: float         = 2e-4,
    lr_g: float | None = None,
    label_dim: int    = 0,
    cat_strategy: str = 'gumbel',
    d_steps: int      = 1,
    g_steps: int      = 1,
    temp_start: float = 1.0,
    temp_end:   float = 0.5,
    verbose_every: int = 50,
    hidden_g: int     = 256,
):
    """
    Train a (conditional) GAN.
    - label_dim=0  → standard unconditional GAN
    - label_dim=2  → cGAN conditioned on binary income label
    Returns (generator, discriminator, history_dict).
    """
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                        shuffle=True, drop_last=True)

    G = Generator(z_dim=z_dim, label_dim=label_dim, hidden=hidden_g,
                  cat_strategy=cat_strategy, temperature=temp_start).to(DEVICE)
    D = Discriminator(data_dim=DATA_DIM, label_dim=label_dim).to(DEVICE)

    opt_G   = optim.Adam(G.parameters(), lr=lr_g or lr, betas=(0.5, 0.999))
    opt_D   = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    crit    = nn.BCELoss()
    history = {'g_loss': [], 'd_loss': []}

    for epoch in range(n_epochs):
        G.temperature = temp_start + (epoch / max(n_epochs - 1, 1)) * (temp_end - temp_start)
        ep_g, ep_d = [], []

        for X_real, y_batch in loader:
            X_real  = X_real.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            b = X_real.size(0)
            real_lbl = torch.full((b,), 0.9, device=DEVICE)   # label smoothing
            fake_lbl = torch.full((b,), 0.1, device=DEVICE)

            y_real_oh = (F.one_hot(y_batch, NUM_CLASSES).float()
                         if label_dim > 0 else None)

            # ── Discriminator updates ──────────────────────────────────────
            for _ in range(d_steps):
                z = torch.randn(b, z_dim, device=DEVICE)
                if label_dim > 0:
                    # Use the real batch's labels for both sides so D must distinguish
                    # real vs fake given the same class label (cGAN spec requirement).
                    X_fake = G(z, y_real_oh).detach()
                    d_r    = D(X_real, y_real_oh)
                    d_f    = D(X_fake, y_real_oh)
                else:
                    X_fake = G(z).detach()
                    d_r    = D(X_real)
                    d_f    = D(X_fake)

                d_loss = crit(d_r, real_lbl) + crit(d_f, fake_lbl)
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
                ep_d.append(d_loss.item())

            # ── Generator updates ──────────────────────────────────────────
            for _ in range(g_steps):
                z = torch.randn(b, z_dim, device=DEVICE)
                if label_dim > 0:
                    # Use the same real-batch labels as the D-update so G is trained
                    # to fool D under the identical conditioning signal D was just
                    # updated on — keeping both steps consistent.
                    X_fake = G(z, y_real_oh)
                    d_f    = D(X_fake, y_real_oh)
                else:
                    X_fake = G(z)
                    d_f    = D(X_fake)

                g_loss = crit(d_f, torch.ones(b, device=DEVICE))
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()
                ep_g.append(g_loss.item())

        history['g_loss'].append(float(np.mean(ep_g)))
        history['d_loss'].append(float(np.mean(ep_d)))

        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"  [{epoch+1:3d}/{n_epochs}]  G={history['g_loss'][-1]:.4f}"
                  f"  D={history['d_loss'][-1]:.4f}  T={G.temperature:.3f}")

    return G, D, history

# %% [markdown]
# ## 6. Synthetic Data Generation

# %%
@torch.no_grad()
def generate_synthetic(
    G: Generator,
    n_samples: int,
    z_dim: int    = 128,
    label_dim: int = 0,
    label_dist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Generate n_samples synthetic records.
    For cGAN supply label_dist (probability per class).
    Returns (X_synth [N, DATA_DIM], y_synth [N] or None).
    """
    G.eval()
    xs, ys = [], []
    done = 0
    while done < n_samples:
        b = min(512, n_samples - done)
        z = torch.randn(b, z_dim, device=DEVICE)
        if label_dim > 0:
            probs   = label_dist if label_dist is not None else np.ones(NUM_CLASSES) / NUM_CLASSES
            y_idx   = np.random.choice(NUM_CLASSES, size=b, p=probs)
            y_t     = torch.tensor(y_idx, device=DEVICE)
            y_oh    = F.one_hot(y_t, NUM_CLASSES).float()
            xs.append(G(z, y_oh).cpu().numpy())
            ys.append(y_idx)
        else:
            xs.append(G(z).cpu().numpy())
        done += b
    G.train()
    X = np.concatenate(xs)[:n_samples]
    y = np.concatenate(ys)[:n_samples] if ys else None
    return X, y

# %% [markdown]
# ## 7. Evaluation Metrics
#
# ### Detection (Section 6e)
# Train a Random Forest on a 50/50 mix of real and synthetic data.
# Use 4-fold cross-validation. **Low AUC ≈ 0.5 is good** (synthetic indistinguishable from real).
#
# ### Efficacy (Section 6f)
# AUC(RF trained on synthetic, tested on real) / AUC(RF trained on real, tested on real).
# **High score ≈ 1 is good** (synthetic is a useful substitute).

# %%
def detection_metric(
    X_real_train: np.ndarray,
    X_synth:      np.ndarray,
) -> tuple[float, float]:
    """
    Detection AUC following the assignment spec (Section 6e):
      - Split real and synthetic into 4 folds each (separately, to maintain balance).
      - Each of 4 rounds: test on 2 folds (1 real + 1 synthetic), train on the
        remaining 6 folds (3 real + 3 synthetic).
      - Report average AUC over 4 rounds.  Low AUC ≈ 0.5 is good.
    """
    N = min(len(X_real_train), len(X_synth))
    Xr = X_real_train[:N]
    Xs = X_synth[:N]

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    # Build fold indices for real and synthetic separately
    real_folds = [te for _, te in kf.split(Xr, np.zeros(N))]
    synt_folds = [te for _, te in kf.split(Xs, np.ones(N))]

    aucs = []
    for i in range(4):
        te_r = real_folds[i]
        te_s = synt_folds[i]
        tr_r = np.concatenate([real_folds[j] for j in range(4) if j != i])
        tr_s = np.concatenate([synt_folds[j] for j in range(4) if j != i])

        X_train = np.concatenate([Xr[tr_r], Xs[tr_s]])
        y_train = np.array([1] * len(tr_r) + [0] * len(tr_s))
        X_test  = np.concatenate([Xr[te_r], Xs[te_s]])
        y_test  = np.array([1] * len(te_r) + [0] * len(te_s))

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def efficacy_metric(
    X_tr_real: np.ndarray, y_tr_real: np.ndarray,
    X_synth:   np.ndarray, y_synth:   np.ndarray,
    X_test:    np.ndarray, y_test:    np.ndarray,
) -> tuple[float, float, float]:
    """Returns (efficacy_ratio, auc_real, auc_synth)."""
    clf_r = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_r.fit(X_tr_real, y_tr_real)
    auc_r = roc_auc_score(y_test, clf_r.predict_proba(X_test)[:, 1])

    clf_s = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_s.fit(X_synth, y_synth.astype(int))
    proba_s = clf_s.predict_proba(X_test)
    if proba_s.shape[1] == 1:
        # Pseudo-labels contained only one class — classifier is degenerate
        auc_s = 0.5
    else:
        auc_s = roc_auc_score(y_test, proba_s[:, list(clf_s.classes_).index(1)])

    return auc_s / max(auc_r, 1e-9), auc_r, auc_s

# %% [markdown]
# ## 8. Visualization Helpers

# %%
def plot_losses(history: dict, title: str = 'GAN Training', save: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history['g_loss'], label='Generator',     alpha=0.85)
    ax.plot(history['d_loss'], label='Discriminator', alpha=0.85)
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
    plt.show(); plt.close()


def plot_distributions(df_real: pd.DataFrame, df_synth: pd.DataFrame,
                       prefix: str = 'dist') -> None:
    # --- Continuous ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i, col in enumerate(CONTINUOUS_COLS):
        ax = axes[i // 3][i % 3]
        ax.hist(df_real[col].dropna(),  bins=40, alpha=0.5, density=True,
                label='Real',      color='steelblue')
        ax.hist(df_synth[col].dropna(), bins=40, alpha=0.5, density=True,
                label='Synthetic', color='coral')
        ax.set_title(col); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle('Continuous Feature Distributions: Real vs Synthetic')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/{prefix}_continuous.png', dpi=100, bbox_inches='tight')
    plt.show(); plt.close()

    # --- Categorical ---
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for i, col in enumerate(CATEGORICAL_COLS):
        ax = axes[i // 4][i % 4]
        r  = df_real[col].value_counts(normalize=True).sort_index()
        s  = df_synth[col].value_counts(normalize=True).sort_index()
        cats = sorted(set(r.index) | set(s.index))
        x    = np.arange(len(cats))
        w    = 0.4
        ax.bar(x - w/2, [r.get(c, 0) for c in cats], w, label='Real',      alpha=0.7, color='steelblue')
        ax.bar(x + w/2, [s.get(c, 0) for c in cats], w, label='Synthetic', alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=5)
        ax.set_title(col, fontsize=9); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle('Categorical Feature Distributions: Real vs Synthetic')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/{prefix}_categorical.png', dpi=100, bbox_inches='tight')
    plt.show(); plt.close()


def plot_correlation(df_real: pd.DataFrame, df_synth: pd.DataFrame,
                     title: str = '', save: str | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cr = df_real[CONTINUOUS_COLS].corr()
    cs = df_synth[CONTINUOUS_COLS].corr()
    kw = dict(annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True,
              linewidths=0.5, annot_kws={'size': 8})
    sns.heatmap(cr, ax=axes[0], **kw); axes[0].set_title('Real')
    sns.heatmap(cs, ax=axes[1], **kw); axes[1].set_title('Synthetic')
    sns.heatmap((cr - cs).abs(), ax=axes[2], cmap='Reds', vmin=0, vmax=1,
                annot=True, fmt='.2f', square=True, linewidths=0.5,
                annot_kws={'size': 8}); axes[2].set_title('|Diff|')
    plt.suptitle(f'Correlation Matrices – {title}')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
    plt.show(); plt.close()

# %% [markdown]
# ## 9. Section 6 – Main Experiments (3 Seeds)
#
# ### Training setup summary
#
# | Hyperparameter | Value |
# |---|---|
# | Seeds | 42, 123, 456 |
# | Train / test split | 80 / 20 (stratified on income) |
# | Epochs | 200 (Sections 6–9); 100 (Section 7); 80 (Section 8) |
# | Batch size | 256 |
# | Noise dimension z | 128 |
# | Learning rate | 2 × 10⁻⁴ (Adam, β₁=0.5, β₂=0.999) |
# | Loss function | Binary cross-entropy (BCE) with one-sided label smoothing (real→0.9, fake→0.1) |
# | Generator | 3-layer MLP (256 hidden), BatchNorm + LeakyReLU(0.2), mixed output head |
# | Discriminator | 3-layer MLP (256 hidden), LeakyReLU(0.2) + Dropout(0.3), no BatchNorm |
# | Categorical strategy | Gumbel-Softmax + straight-through estimator (default); Softmax (Section 7 ablation) |
# | Temperature schedule | Linear anneal 1.0 → 0.5 over training |
# | D : G update ratio | 1:1 (default); 5:1 (Section 8 collapse experiment) |
#
# For each seed:
# 1. Stratified 80/20 train–test split.
# 2. Train standard GAN (unconditional) for 200 epochs.
# 3. Train cGAN (conditioned on income) for 200 epochs.
# 4. Generate synthetic dataset of size equal to the training set.
# 5. Evaluate **detection** (4-fold CV AUC) and **efficacy** (AUC ratio).

# %%
SEEDS      = [42, 123, 456]
Z_DIM      = 128
N_EPOCHS   = 200
BATCH_SIZE = 256

results = {'gan':  {'detection': [], 'efficacy': []},
           'cgan': {'detection': [], 'efficacy': []}}

# %%
for si, seed in enumerate(SEEDS):
    print(f"\n{'='*65}\n  Experiment {si+1}/3  |  seed={seed}\n{'='*65}")
    set_seed(seed)

    df_tr, df_te = train_test_split(df, test_size=0.2, random_state=seed,
                                    stratify=df[TARGET_COL])
    X_tr, y_tr, sc = encode_df(df_tr)
    X_te, y_te, _  = encode_df(df_te, scaler=sc)
    label_dist = df_tr[TARGET_COL].value_counts(normalize=True).sort_index().values
    print(f"Train={len(df_tr)}  Test={len(df_te)}  "
          f"label_dist={np.round(label_dist, 3)}")

    # ── Standard GAN ──────────────────────────────────────────────────
    print("\n[GAN] Training …")
    G_gan, D_gan, hist_gan = train_gan(
        X_tr, y_tr, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        z_dim=Z_DIM, lr=2e-4, label_dim=0, cat_strategy='gumbel')

    plot_losses(hist_gan, title=f'GAN Loss (seed={seed})',
                save=f'{FIGURES_DIR}/gan_loss_seed{seed}.png')

    X_synth_gan, _ = generate_synthetic(G_gan, len(X_tr), z_dim=Z_DIM, label_dim=0)

    # Pseudo-label synthetic samples using a RF trained on real data.
    # Randomly shuffling labels (previous approach) produces labels uncorrelated
    # with the generated features, making the efficacy metric meaningless.
    _clf_pseudo = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    _clf_pseudo.fit(X_tr, y_tr.astype(int))
    y_synth_gan = _clf_pseudo.predict(X_synth_gan).astype(np.float32)

    det_gan, det_std = detection_metric(X_tr, X_synth_gan)
    eff_gan, auc_r, auc_sg = efficacy_metric(X_tr, y_tr, X_synth_gan, y_synth_gan, X_te, y_te)
    results['gan']['detection'].append(det_gan)
    results['gan']['efficacy'].append(eff_gan)
    print(f"  Detection AUC: {det_gan:.4f} ± {det_std:.4f}  (↓ is better)")
    print(f"  Efficacy:      {eff_gan:.4f}  (real={auc_r:.4f}, synth={auc_sg:.4f})")

    # ── Conditional GAN ───────────────────────────────────────────────
    print("\n[cGAN] Training …")
    G_cgan, D_cgan, hist_cgan = train_gan(
        X_tr, y_tr, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        z_dim=Z_DIM, lr=2e-4, label_dim=NUM_CLASSES, cat_strategy='gumbel')

    plot_losses(hist_cgan, title=f'cGAN Loss (seed={seed})',
                save=f'{FIGURES_DIR}/cgan_loss_seed{seed}.png')

    X_synth_cgan, y_synth_cgan = generate_synthetic(
        G_cgan, len(X_tr), z_dim=Z_DIM, label_dim=NUM_CLASSES, label_dist=label_dist)

    det_cgan, det_std_c = detection_metric(X_tr, X_synth_cgan)
    eff_cgan, _, auc_sc = efficacy_metric(X_tr, y_tr, X_synth_cgan, y_synth_cgan, X_te, y_te)
    results['cgan']['detection'].append(det_cgan)
    results['cgan']['efficacy'].append(eff_cgan)
    print(f"  Detection AUC: {det_cgan:.4f} ± {det_std_c:.4f}  (↓ is better)")
    print(f"  Efficacy:      {eff_cgan:.4f}  (real={auc_r:.4f}, synth={auc_sc:.4f})")

    # Visualise feature distributions for the first seed only
    if si == 0:
        df_real_dec  = decode_to_df(X_tr,         sc)
        df_real_dec[TARGET_COL] = y_tr.astype(int)
        df_synth_gan_dec  = decode_to_df(X_synth_gan,  sc)
        df_synth_cgan_dec = decode_to_df(X_synth_cgan, sc)
        plot_distributions(df_real_dec, df_synth_gan_dec,  prefix='gan_dist')
        plot_distributions(df_real_dec, df_synth_cgan_dec, prefix='cgan_dist')
        plot_correlation(df_real_dec, df_synth_gan_dec,
                         title='GAN', save=f'{FIGURES_DIR}/gan_corr.png')
        plot_correlation(df_real_dec, df_synth_cgan_dec,
                         title='cGAN', save=f'{FIGURES_DIR}/cgan_corr.png')

# %%
print("\n" + "="*65)
print("  SUMMARY – averaged over 3 seeds")
print("="*65)
for model in ('gan', 'cgan'):
    d = results[model]['detection']
    e = results[model]['efficacy']
    print(f"\n  {model.upper()}")
    print(f"    Detection AUC : {np.mean(d):.4f} ± {np.std(d):.4f}  (↓ better)")
    print(f"    Efficacy      : {np.mean(e):.4f} ± {np.std(e):.4f}  (↑ better)")

# %% [markdown]
# ## 10. Section 7 – Discrete Features: Gradient Flow & Representation
#
# ### Why argmax breaks gradient flow
# Feeding argmax-decoded one-hot vectors into the discriminator prevents the generator
# from learning because **argmax is not differentiable**. More precisely, the
# sub-gradient of argmax is zero almost everywhere: for any logit vector $\mathbf{l}$,
# the one-hot output is a constant with respect to small perturbations of $\mathbf{l}$
# (it only changes at the isolated points where the maximum switches). Consequently,
# $\partial\text{loss}/\partial\mathbf{l} = 0$ everywhere except at measure-zero
# switching boundaries. Backpropagation through the discriminator reaches the
# argmax → gradient is zero → the generator parameters receive no useful update signal
# and cannot learn.
#
# ### Strategy 1 – Plain Softmax (forward & backward)
# **Forward**: generator outputs $\text{softmax}(\mathbf{l}/T)$ — a continuous
# probability vector in $\Delta^{K-1}$. No hard decision; the discriminator sees
# soft probabilities.
# **Backward**: $\partial\text{softmax}/\partial\mathbf{l}$ is a well-defined
# $(K\!\times\!K)$ Jacobian, so gradients flow. However the discriminator now sees
# real data (hard 0/1) vs. fake data (soft 0…1) — a format mismatch that can
# confuse training.
#
# ### Strategy 2 – Gumbel-Softmax + Straight-Through Estimator (forward & backward)
# **Forward**: Gumbel noise is added to the logits, then a hard one-hot is taken via
# $\text{argmax}$. The discriminator sees (near-)discrete samples identical in format
# to the real data → no distributional mismatch.
# **Backward**: instead of using the zero gradient of argmax, we substitute it with
# the gradient of the soft Gumbel-Softmax function (the ST trick:
# `y_hard - y_soft.detach() + y_soft`). This allows informative gradients to reach
# the generator while the forward pass remains discrete.

# %%
print("Section 7: Softmax vs Gumbel-Softmax comparison")
set_seed(42)
df_tr7, df_te7 = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df[TARGET_COL])
X_tr7, y_tr7, sc7 = encode_df(df_tr7)
X_te7, y_te7, _   = encode_df(df_te7, scaler=sc7)

N_EP7 = 100   # fewer epochs — only need visible trends

print("\n[Softmax strategy] Training …")
G_soft, _, hist_soft = train_gan(
    X_tr7, y_tr7, n_epochs=N_EP7, batch_size=BATCH_SIZE,
    z_dim=Z_DIM, lr=2e-4, label_dim=0, cat_strategy='softmax', verbose_every=50)

print("\n[Gumbel-ST strategy] Training …")
G_gum,  _, hist_gum  = train_gan(
    X_tr7, y_tr7, n_epochs=N_EP7, batch_size=BATCH_SIZE,
    z_dim=Z_DIM, lr=2e-4, label_dim=0, cat_strategy='gumbel',  verbose_every=50)

# %%
# Loss curve comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, hist, name in zip(axes, [hist_soft, hist_gum], ['Softmax', 'Gumbel-ST']):
    ax.plot(hist['g_loss'], label='G loss')
    ax.plot(hist['d_loss'], label='D loss')
    ax.set_title(f'{name} – Loss Curves')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/s7_loss_comparison.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

# %%
def cat_entropy_per_feature(G: Generator, n: int = 3000, z_dim: int = 128) -> dict[str, float]:
    """
    Average entropy of generated soft categorical distributions (from raw logits,
    without Gumbel noise) — measures how 'committed' the generator is.
    """
    G.eval()
    with torch.no_grad():
        z = torch.randn(n, z_dim, device=DEVICE)
        h = G.body(z)
        out = {}
        for col, head in zip(CATEGORICAL_COLS, G.cat_heads):
            probs = F.softmax(head(h), dim=-1).cpu().numpy()
            H     = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            out[col] = float(np.mean(H))
    G.train()
    return out


def real_entropy(df_tr: pd.DataFrame) -> dict[str, float]:
    """Shannon entropy of the empirical marginal for each categorical feature."""
    out = {}
    for col in CATEGORICAL_COLS:
        p = df_tr[col].value_counts(normalize=True).values
        out[col] = float(-np.sum(p * np.log(p + 1e-10)))
    return out

# %%
ent_soft  = cat_entropy_per_feature(G_soft, z_dim=Z_DIM)
ent_gum   = cat_entropy_per_feature(G_gum,  z_dim=Z_DIM)
ent_real  = real_entropy(df_tr7)

print(f"\n{'Feature':<20} {'Real':>8} {'Softmax':>10} {'Gumbel-ST':>10}")
print("-" * 52)
for col in CATEGORICAL_COLS:
    print(f"{col:<20} {ent_real[col]:>8.3f} {ent_soft[col]:>10.3f} {ent_gum[col]:>10.3f}")

# %%
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(CATEGORICAL_COLS))
w = 0.25
ax.bar(x - w,   [ent_real[c] for c in CATEGORICAL_COLS], w, label='Real data',  color='dimgray', alpha=0.85)
ax.bar(x,       [ent_soft[c] for c in CATEGORICAL_COLS], w, label='Softmax',    color='steelblue', alpha=0.85)
ax.bar(x + w,   [ent_gum[c]  for c in CATEGORICAL_COLS], w, label='Gumbel-ST',  color='coral',   alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(CATEGORICAL_COLS, rotation=30, ha='right')
ax.set_ylabel('Avg entropy (nats)')
ax.set_title('Section 7 – Categorical Feature Entropy: Real vs Generated')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/s7_entropy.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

# %%
# Efficacy comparison for the two strategies
X_synth_soft7, _ = generate_synthetic(G_soft, len(X_tr7), z_dim=Z_DIM)
X_synth_gum7,  _ = generate_synthetic(G_gum,  len(X_tr7), z_dim=Z_DIM)

_clf_pseudo7 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
_clf_pseudo7.fit(X_tr7, y_tr7.astype(int))
y_s7_soft = _clf_pseudo7.predict(X_synth_soft7).astype(np.float32)
y_s7_gum  = _clf_pseudo7.predict(X_synth_gum7).astype(np.float32)

det_soft7, _ = detection_metric(X_tr7, X_synth_soft7)
det_gum7,  _ = detection_metric(X_tr7, X_synth_gum7)
eff_soft7, _, _ = efficacy_metric(X_tr7, y_tr7, X_synth_soft7, y_s7_soft, X_te7, y_te7)
eff_gum7,  _, _ = efficacy_metric(X_tr7, y_tr7, X_synth_gum7,  y_s7_gum,  X_te7, y_te7)

print(f"\nSoftmax   – Detection: {det_soft7:.4f}  Efficacy: {eff_soft7:.4f}")
print(f"Gumbel-ST – Detection: {det_gum7:.4f}  Efficacy: {eff_gum7:.4f}")

# %% [markdown]
# **Summary (Section 7):**
# - Gumbel-ST entropy should be closer to the real-data entropy than Softmax,
#   indicating the generator is more "committed" to realistic category distributions.
# - Softmax generators often emit near-uniform distributions for large-vocabulary
#   features (high entropy), relying on argmax post-hoc rather than learning true
#   categorical structure — evidenced by entropy >> real-data entropy.
# - Based on the entropy comparison, fidelity metrics, and training stability,
#   Gumbel-ST is the preferred representation because it (a) eliminates the
#   real/fake format mismatch seen by the discriminator, and (b) produces
#   informative, differentiable gradients.

# %% [markdown]
# ## 11. Section 8 – Mode Collapse: Induce, Diagnose, Mitigate
#
# ### Collapse indicator
# We use the **categorical coverage ratio**:
# $$\text{coverage} = \frac{|\text{unique }(\text{workclass}, \text{marital-status},
#   \text{occupation})\text{ combos in synthetic}|}
#  {|\text{unique combos in real training set}|}$$
# A value near 0 indicates the generator maps all noise inputs to a narrow set of
# discrete patterns (mode collapse). A value near 1 indicates full diversity.
# We restrict to three key categorical features to keep distinct-combo counts
# tractable yet meaningful.
#
# ### Inducing collapse — revised strategy
# The earlier strategy (10:1 D:G ratio + slow lr_g=2e-5) failed to produce
# under-diversity because **Gumbel-ST noise creates a diversity floor**: with
# lr_g=2e-5 the generator's weights barely update and remain near random
# initialization, so Gumbel noise fully controls categorical sampling, producing
# near-maximum diversity regardless of discriminator pressure (coverage 2.144 —
# over-diversity, not collapse).
#
# The revised strategy removes the diversity floor directly:
# 1. **Sharply reduce generator capacity** (hidden_g=16, down from 256) — the
#    bottleneck prevents the generator from expressing the full categorical
#    distribution and forces it to concentrate probability mass on a few modes.
# 2. **Switch to plain softmax** — eliminates Gumbel noise so there is no
#    randomness-driven diversity floor; the generator's learned distribution
#    fully determines the output.
# 3. **5:1 D:G update ratio** — discriminator stays ahead, accelerating collapse.

# %%
COLLAPSE_COLS = ['workclass', 'marital-status', 'occupation']


def coverage_ratio(df_synth_dec: pd.DataFrame, df_real_dec: pd.DataFrame) -> tuple[float, int, int]:
    """Categorical coverage ratio for COLLAPSE_COLS."""
    real_combos  = set(zip(*[df_real_dec[c].astype(str)  for c in COLLAPSE_COLS]))
    synth_combos = set(zip(*[df_synth_dec[c].astype(str) for c in COLLAPSE_COLS]))
    ratio = len(synth_combos) / max(len(real_combos), 1)
    return ratio, len(real_combos), len(synth_combos)

# %%
print("Section 8: Mode Collapse Experiments")
set_seed(42)
df_tr8, df_te8 = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df[TARGET_COL])
X_tr8, y_tr8, sc8 = encode_df(df_tr8)
df_real8_dec = decode_to_df(X_tr8, sc8)

N_EP8 = 80  # short runs; trends are visible even with fewer epochs

# ── 8a: Baseline (1:1 D:G updates) ────────────────────────────────────
print("\n[Baseline 1:1] Training …")
G_b8, _, hist_b8 = train_gan(
    X_tr8, y_tr8, n_epochs=N_EP8, batch_size=BATCH_SIZE,
    z_dim=Z_DIM, lr=2e-4, label_dim=0, cat_strategy='gumbel',
    d_steps=1, g_steps=1, verbose_every=40)

# ── 8b: Induced collapse (reduced capacity + softmax + 5:1 D:G) ───────
# Removing Gumbel noise eliminates the diversity floor; the bottlenecked
# generator (hidden=16) cannot express the full categorical distribution.
print("\n[Collapsed: hidden=16, softmax, 5:1 D:G] Training …")
G_c8, _, hist_c8 = train_gan(
    X_tr8, y_tr8, n_epochs=N_EP8, batch_size=BATCH_SIZE,
    z_dim=Z_DIM, lr=2e-4, lr_g=2e-4, label_dim=0, cat_strategy='softmax',
    d_steps=5, g_steps=1, hidden_g=16, verbose_every=40)

# %%
Xsb8, _ = generate_synthetic(G_b8, len(X_tr8), z_dim=Z_DIM)
Xsc8, _ = generate_synthetic(G_c8, len(X_tr8), z_dim=Z_DIM)

cov_b8, n_real8, n_synth_b8 = coverage_ratio(decode_to_df(Xsb8, sc8), df_real8_dec)
cov_c8, _,       n_synth_c8 = coverage_ratio(decode_to_df(Xsc8, sc8), df_real8_dec)

print(f"\nReal  unique combos: {n_real8}")
print(f"Baseline  coverage : {cov_b8:.4f}  ({n_synth_b8}/{n_real8})")
print(f"Collapsed coverage : {cov_c8:.4f}  ({n_synth_c8}/{n_real8})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, hist, t in zip(axes, [hist_b8, hist_c8],
                       ['Baseline (1:1, hidden=256, Gumbel)',
                        'Collapsed (5:1, hidden=16, Softmax)']):
    ax.plot(hist['g_loss'], label='G loss')
    ax.plot(hist['d_loss'], label='D loss')
    ax.set_title(t); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Section 8 – Mode Collapse Loss Curves')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/s8_collapse_loss.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

# %% [markdown]
# ### 8c–d – Mitigation: Minibatch Standard Deviation
#
# **Prediction (written before running):** The MBD layer appends the mean
# per-feature standard deviation across the mini-batch to the discriminator's
# penultimate representation. When the bottlenecked generator collapses (all fake
# samples look similar), the batch std of fake samples is near zero, making them
# trivially detectable regardless of whether any individual sample looks plausible.
# Faced with this extra signal, the generator must spread its outputs to survive.
# We therefore predict:
# - Coverage ratio recovers toward the baseline level (> collapsed value).
# - G-loss stabilises at a lower value than the collapsed run (generator gets
#   useful gradient again).
# - D-loss does not drop as far as in the collapsed run (diversity signal prevents
#   total discriminator dominance).

# %%
class DiscriminatorMBD(nn.Module):
    """Discriminator with Minibatch Standard Deviation layer."""
    def __init__(self, data_dim: int = DATA_DIM, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, hidden),   nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(hidden // 2 + 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        h    = self.body(x)
        # Minibatch std: mean std across features, broadcast to batch
        mstd = h.std(dim=0, keepdim=True).mean(dim=1, keepdim=True).expand(h.size(0), 1)
        return self.head(torch.cat([h, mstd], dim=1)).squeeze(1)


def train_with_mbd(X_train, y_train, n_epochs=80, batch_size=256, z_dim=128,
                   lr=2e-4, lr_g=None, d_steps=5, hidden_g=16,
                   cat_strategy='softmax'):
    """Train GAN with MBD discriminator and intentional D:G imbalance.

    Uses the same collapse settings (hidden_g=16, softmax) so the mitigation
    is tested on the same type of collapse that was induced.
    """
    loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(z_dim=z_dim, label_dim=0, hidden=hidden_g,
                  cat_strategy=cat_strategy, temperature=1.0).to(DEVICE)
    D = DiscriminatorMBD().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=lr_g or lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    crit  = nn.BCELoss()
    hist  = {'g_loss': [], 'd_loss': []}

    for epoch in range(n_epochs):
        G.temperature = 1.0 + (epoch / max(n_epochs-1, 1)) * (0.5 - 1.0)
        ep_g, ep_d = [], []
        for X_real, _ in loader:
            X_real = X_real.to(DEVICE); b = X_real.size(0)
            rl = torch.full((b,), 0.9, device=DEVICE)
            fl = torch.full((b,), 0.1, device=DEVICE)
            for _ in range(d_steps):
                z = torch.randn(b, z_dim, device=DEVICE)
                d_loss = crit(D(X_real), rl) + crit(D(G(z).detach()), fl)
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
                ep_d.append(d_loss.item())
            z = torch.randn(b, z_dim, device=DEVICE)
            g_loss = crit(D(G(z)), torch.ones(b, device=DEVICE))
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            ep_g.append(g_loss.item())
        hist['g_loss'].append(float(np.mean(ep_g)))
        hist['d_loss'].append(float(np.mean(ep_d)))
        if (epoch + 1) % 40 == 0:
            print(f"  [{epoch+1}/{n_epochs}] G={hist['g_loss'][-1]:.4f}  D={hist['d_loss'][-1]:.4f}")
    return G, hist

# %%
print("\n[MBD mitigation: 5:1, hidden=16, softmax + MBD] Training …")
G_mbd, hist_mbd = train_with_mbd(X_tr8, y_tr8, n_epochs=N_EP8, d_steps=5,
                                  hidden_g=16, cat_strategy='softmax')

Xsmbd, _ = generate_synthetic(G_mbd, len(X_tr8), z_dim=Z_DIM)
cov_mbd, _, n_synth_mbd = coverage_ratio(decode_to_df(Xsmbd, sc8), df_real8_dec)

print(f"\nMBD mitigation coverage: {cov_mbd:.4f}  ({n_synth_mbd}/{n_real8})")
print(f"(Baseline={cov_b8:.4f}  Collapsed={cov_c8:.4f}  MBD={cov_mbd:.4f})")

# %% [markdown]
# ### 8d – Reconciliation
#
# **Predicted:** The MBD layer would push coverage back toward baseline (> collapsed)
# and prevent D-loss from collapsing to ~0, because the generator is penalised the
# moment its outputs become homogeneous (low batch std).
#
# **Observed:** See printed coverage values and loss curves above.
#
# **Interpretation:** If `cov_mbd > cov_c8` the prediction is confirmed: the batch
# std signal forced the bottlenecked generator to spread its outputs across more
# combinations to avoid easy detection. G-loss and D-loss should both be less extreme
# than in the collapsed run. The MBD technique targets the root cause — D cannot see
# diversity across the batch — rather than patching symptoms, which is why it works
# even when the generator has severely limited capacity.
# If `cov_mbd ≈ cov_c8`, the 5:1 D:G imbalance may still be too strong; a lighter
# ratio (2:1 or 3:1) would isolate MBD's effect more cleanly.

# %%
# Summary bar chart
fig, ax = plt.subplots(figsize=(8, 4))
labels  = ['Baseline\n(1:1, 256, Gumbel)', 'Collapsed\n(5:1, 16, Softmax)', 'MBD Mitigated\n(5:1, 16, Softmax+MBD)']
covs    = [cov_b8, cov_c8, cov_mbd]
colors  = ['steelblue', 'crimson', 'seagreen']
bars    = ax.bar(labels, covs, color=colors, alpha=0.85, width=0.5)
ax.axhline(1.0, color='gray', linestyle='--', label='Perfect coverage')
for bar, val in zip(bars, covs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
            f'{val:.3f}', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
ax.set_ylabel('Categorical Coverage Ratio'); ax.set_ylim(0, max(covs) * 1.15)
ax.set_title('Section 8 – Mode Collapse Coverage Comparison')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/s8_coverage.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

# Loss comparison: all three
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, hist, t in zip(axes, [hist_b8, hist_c8, hist_mbd],
                        ['Baseline (1:1, 256, Gumbel)',
                         'Collapsed (5:1, 16, Softmax)',
                         'MBD (5:1, 16, Softmax+MBD)']):
    ax.plot(hist['g_loss'], label='G loss')
    ax.plot(hist['d_loss'], label='D loss')
    ax.set_title(t); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Section 8 – Loss Curves: Baseline / Collapse / Mitigation')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/s8_all_loss.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

# %% [markdown]
# ## 12. Section 9 – Open-Ended Contribution: Spectral Normalization
#
# ### Modification
# Apply **spectral normalization (SN)** to every linear layer of the discriminator.
#
# ### Motivation (first principles)
# The standard GAN discriminator has no explicit constraint on its Lipschitz constant.
# A large Lipschitz constant means the loss landscape has steep gradients that push
# the generator toward sharp, unstable updates; when the discriminator becomes too
# confident early, the generator's gradients saturate (sigmoid → 0) and learning
# stalls. Spectral normalization constrains the spectral norm (largest singular value)
# of each weight matrix $\mathbf{W}$ to $\le 1$ by dividing by
# $\sigma_1(\mathbf{W})$ at each forward pass, bounding the overall Lipschitz
# constant of the discriminator to 1. This targets the root cause of gradient
# saturation without requiring an extra gradient-penalty term or architectural changes.
#
# ### Explicit prediction (written before running the experiment)
# Spectral normalization should:
# 1. **Reduce G-loss variance** (measured by $\text{Var}(\text{G-loss curve})$) —
#    because bounding D's Lipschitz constant prevents the sharp confidence spikes
#    that cause the generator's gradient to oscillate.
# 2. **Prevent D-loss from collapsing to near-zero** — the constrained D cannot
#    become arbitrarily confident, so it keeps providing a useful learning signal.
# 3. **Improve or maintain detection AUC** (closer to 0.5) and efficacy ratio —
#    more stable training should yield a generator that better covers the data manifold.
#
# These are falsifiable: if G-loss variance *increases* or D-loss still collapses,
# the prediction is wrong and must be diagnosed below.

# %% [markdown]
# ### Implementation

# %%
class DiscriminatorSN(nn.Module):
    """Discriminator with Spectral Normalization on all linear layers."""
    def __init__(self, data_dim: int = DATA_DIM, hidden: int = 256):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Linear(data_dim, hidden)), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            sn(nn.Linear(hidden, hidden)),   nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            sn(nn.Linear(hidden, hidden//2)),nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Linear(hidden//2, 1)),     nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_gan_sn(X_train, y_train, n_epochs=200, batch_size=256, z_dim=128, lr=2e-4):
    """Train GAN with spectral-norm discriminator."""
    loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(z_dim=z_dim, label_dim=0, cat_strategy='gumbel').to(DEVICE)
    D = DiscriminatorSN().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    crit  = nn.BCELoss()
    hist  = {'g_loss': [], 'd_loss': []}

    for epoch in range(n_epochs):
        G.temperature = 1.0 + (epoch / max(n_epochs-1, 1)) * (0.5 - 1.0)
        ep_g, ep_d = [], []
        for X_real, _ in loader:
            X_real = X_real.to(DEVICE); b = X_real.size(0)
            rl = torch.full((b,), 0.9, device=DEVICE)
            fl = torch.full((b,), 0.1, device=DEVICE)
            z = torch.randn(b, z_dim, device=DEVICE)
            d_loss = crit(D(X_real), rl) + crit(D(G(z).detach()), fl)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            ep_d.append(d_loss.item())
            z = torch.randn(b, z_dim, device=DEVICE)
            g_loss = crit(D(G(z)), torch.ones(b, device=DEVICE))
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            ep_g.append(g_loss.item())
        hist['g_loss'].append(float(np.mean(ep_g)))
        hist['d_loss'].append(float(np.mean(ep_d)))
        if (epoch + 1) % 50 == 0:
            print(f"  [{epoch+1}/{n_epochs}] G={hist['g_loss'][-1]:.4f}  D={hist['d_loss'][-1]:.4f}")
    return G, D, hist

# %%
print("Section 9: Spectral Normalization Experiment")
set_seed(42)
df_tr9, df_te9 = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df[TARGET_COL])
X_tr9, y_tr9, sc9 = encode_df(df_tr9)
X_te9, y_te9, _   = encode_df(df_te9, scaler=sc9)

print("\n[Baseline GAN] Training …")
G_b9, _, hist_b9 = train_gan(
    X_tr9, y_tr9, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
    z_dim=Z_DIM, lr=2e-4, label_dim=0, cat_strategy='gumbel')

print("\n[Spectral-Norm GAN] Training …")
G_sn, D_sn, hist_sn = train_gan_sn(
    X_tr9, y_tr9, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, z_dim=Z_DIM, lr=2e-4)

# %%
# Loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, hist, t in zip(axes, [hist_b9, hist_sn], ['Baseline GAN', 'Spectral-Norm GAN']):
    ax.plot(hist['g_loss'], label='G loss', alpha=0.85)
    ax.plot(hist['d_loss'], label='D loss', alpha=0.85)
    ax.set_title(t); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Section 9 – Spectral Normalization vs Baseline')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/s9_sn_loss.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()

# %%
# G-loss variance comparison (prediction check)
var_b9 = float(np.var(hist_b9['g_loss']))
var_sn = float(np.var(hist_sn['g_loss']))
print(f"\nG-loss variance  Baseline: {var_b9:.6f}   Spectral-Norm: {var_sn:.6f}")
print(f"Relative change: {(var_b9 - var_sn) / max(var_b9, 1e-9) * 100:.1f}%  "
      f"({'reduced ✓' if var_sn < var_b9 else 'increased ✗'})")

# Evaluation metrics
Xs_b9, _ = generate_synthetic(G_b9, len(X_tr9), z_dim=Z_DIM)
Xs_sn, _ = generate_synthetic(G_sn, len(X_tr9), z_dim=Z_DIM)

_clf_pseudo9 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
_clf_pseudo9.fit(X_tr9, y_tr9.astype(int))
y_syn9_b = _clf_pseudo9.predict(Xs_b9).astype(np.float32)
y_syn9_sn = _clf_pseudo9.predict(Xs_sn).astype(np.float32)

det_b9, _ = detection_metric(X_tr9, Xs_b9)
det_sn, _ = detection_metric(X_tr9, Xs_sn)
eff_b9, _, _ = efficacy_metric(X_tr9, y_tr9, Xs_b9, y_syn9_b,  X_te9, y_te9)
eff_sn, _, _ = efficacy_metric(X_tr9, y_tr9, Xs_sn, y_syn9_sn, X_te9, y_te9)

print(f"\nBaseline     – Detection AUC: {det_b9:.4f}  Efficacy: {eff_b9:.4f}")
print(f"Spectral-Norm– Detection AUC: {det_sn:.4f}  Efficacy: {eff_sn:.4f}")

# %% [markdown]
# ### Reconciliation
# If G-loss variance decreases, the prediction is confirmed: bounding the
# Lipschitz constant of the discriminator stabilises the gradient signal reaching
# the generator. A detection AUC closer to 0.5 would confirm better synthetic-data
# quality. If efficacy improves, the generator learned a more informative
# representation of the data distribution.
#
# Even if results are mixed, spectral normalization is theoretically sound:
# it costs nothing at inference, introduces no hyperparameters, and addresses the
# root cause of gradient saturation in the discriminator rather than patching
# symptoms.

# %%
print("\n" + "="*65)
print("  ALL EXPERIMENTS COMPLETE")
print("="*65)
print(f"\nFigures saved to: {FIGURES_DIR}/")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  {f}")
