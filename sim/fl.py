"""Multimodal federated learning model (Sec. II-D).

Task: FashionMNIST classification decomposed into three synthetic sensing
modalities per sample:
  r=0 "camera": top half of the image      (14x28)
  r=1 "lidar" : bottom half of the image   (14x28)
  r=2 "radar" : 7x7 average-pooled intensity map

Each vehicle holds a non-IID (Dirichlet) local dataset with heterogeneous
size, a subset of modalities, and a sensing-quality score Q in [0,1]
(low quality = additive noise, modeling night/rain drives). The model has
one small CNN encoder per modality plus a local fusion head that averages
available modality embeddings.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MOD = 3
EMB_DIM = 64
N_CLASS = 10
# over-the-air encoder sizes (bits): camera > lidar > radar, FP32 weights of
# typical perception backbones (ResNet-ish / PointPillars-ish / small radar net)
ENC_BITS = {0: 240e6, 1: 160e6, 2: 80e6}

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)


# ---------------------------------------------------------------- dataset
def load_uci_har(root="results/data"):
    """UCI-HAR raw inertial signals: 3 modalities x 3 axes x 128 steps.

    Modalities: r=0 body_acc, r=1 body_gyro, r=2 total_acc (as in MFedMC,
    Yuan et al., IEEE TMC 2026, natural-distribution setting).
    Returns xtr [N,9,128], ytr, subj_tr, xte [M,9,128], yte.
    """
    import os as _os
    base = _os.path.join(root, "UCI HAR Dataset")
    sigs = ["body_acc", "body_gyro", "total_acc"]

    def load_split(split):
        chans = []
        for s in sigs:
            for ax in "xyz":
                f = _os.path.join(base, split, "Inertial Signals",
                                  f"{s}_{ax}_{split}.txt")
                chans.append(np.loadtxt(f, dtype=np.float32))
        x = torch.tensor(np.stack(chans, axis=1))  # [N, 9, 128]
        y = torch.tensor(
            np.loadtxt(_os.path.join(base, split, f"y_{split}.txt"),
                       dtype=np.int64) - 1)
        subj = np.loadtxt(_os.path.join(base, split, f"subject_{split}.txt"),
                          dtype=np.int64)
        return x, y, subj

    xtr, ytr, subj_tr = load_split("train")
    xte, yte, _ = load_split("test")
    # per-channel standardization (train statistics)
    mu = xtr.mean(dim=(0, 2), keepdim=True)
    sd = xtr.std(dim=(0, 2), keepdim=True) + 1e-6
    return (xtr - mu) / sd, ytr, subj_tr, (xte - mu) / sd, yte


def har_views(x):
    """Split [B,9,128] into the three modality views [B,3,128]."""
    return {0: x[:, 0:3], 1: x[:, 3:6], 2: x[:, 6:9]}


def partition_har(subj_tr, n_veh, seed):
    """Natural non-IID partition: subjects -> vehicles (split into shards)."""
    rng = np.random.RandomState(seed)
    subjects = np.unique(subj_tr)
    shards = []
    per = int(np.ceil(n_veh / len(subjects)))
    for s in subjects:
        idx = np.where(subj_tr == s)[0]
        rng.shuffle(idx)
        shards.extend(np.array_split(idx, per))
    rng.shuffle(shards)
    return [shards[i % len(shards)] for i in range(n_veh)]


class Encoder1D(nn.Module):
    """Small 1D-CNN encoder for inertial time series [B,3,128]."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, 7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.fc = nn.Linear(32 * 8, EMB_DIM)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 4)
        x = F.max_pool1d(F.relu(self.conv2(x)), 4)
        return F.relu(self.fc(x.flatten(1)))


def load_fashion_mnist(root="results/data"):
    from torchvision import datasets, transforms

    tr = datasets.FashionMNIST(root, train=True, download=True)
    te = datasets.FashionMNIST(root, train=False, download=True)
    xtr = tr.data.float().unsqueeze(1) / 255.0
    xte = te.data.float().unsqueeze(1) / 255.0
    return xtr, tr.targets.clone(), xte, te.targets.clone()


def modality_views(x):
    """Split a batch of images [B,1,28,28] into the three modality views."""
    top = x[:, :, :14, :]
    bot = x[:, :, 14:, :]
    radar = F.avg_pool2d(x, 4)  # [B,1,7,7]
    return {0: top, 1: bot, 2: radar}


def partition(y_train, n_veh, seed, dirichlet=0.5, size_lognorm=0.6,
              mean_size=420, p_mod=0.72, q_low_frac=0.35):
    """Non-IID Dirichlet partition with heterogeneous sizes/modalities/quality.

    Returns per-vehicle: sample indices, modality set, per-modality kept
    fraction, quality score per modality.
    """
    rng = np.random.RandomState(seed)
    n_total = len(y_train)
    # vehicle sizes (log-normal)
    sizes = np.clip(
        rng.lognormal(np.log(mean_size), size_lognorm, n_veh), 60, 1800
    ).astype(int)
    # Dirichlet class mixture per vehicle
    by_class = [np.where(y_train.numpy() == c)[0] for c in range(N_CLASS)]
    for c in range(N_CLASS):
        rng.shuffle(by_class[c])
    ptr = [0] * N_CLASS
    veh_idx = []
    for i in range(n_veh):
        mix = rng.dirichlet([dirichlet] * N_CLASS)
        take = (mix * sizes[i]).astype(int)
        idx = []
        for c in range(N_CLASS):
            k = min(take[c], len(by_class[c]) - ptr[c])
            idx.extend(by_class[c][ptr[c]: ptr[c] + k])
            ptr[c] = (ptr[c] + k) % max(len(by_class[c]) - 1, 1)
        veh_idx.append(np.array(idx))

    mods, mod_frac, quality = assign_heterogeneity(n_veh, rng, p_mod,
                                                   q_low_frac)
    return veh_idx, mods, mod_frac, quality


def assign_heterogeneity(n_veh, rng, p_mod=0.72, q_low_frac=0.35):
    """Random modality subsets, per-modality availability, sensing quality."""
    mods, mod_frac, quality = [], [], []
    for i in range(n_veh):
        m = [r for r in range(N_MOD) if rng.rand() < p_mod]
        if not m:
            m = [int(rng.randint(N_MOD))]
        mods.append(sorted(m))
        # per-modality data availability (some vehicles have very little
        # data for one of their modalities -> high learning need)
        fr = {r: float(np.clip(rng.beta(2.0, 1.2), 0.05, 1.0)) for r in m}
        mod_frac.append(fr)
        # quality: a fraction of drives are "night/rain" with degraded sensing
        q = {}
        for r in m:
            q[r] = float(rng.uniform(0.25, 0.55)) if rng.rand() < q_low_frac \
                else float(rng.uniform(0.8, 1.0))
        quality.append(q)
    return mods, mod_frac, quality


# ---------------------------------------------------------------- models
class Encoder(nn.Module):
    """Small CNN encoder shared architecture across modalities."""

    def __init__(self, in_hw):
        super().__init__()
        h, w = in_hw
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * (h // 4) * (w // 4), EMB_DIM)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return F.relu(self.fc(x.flatten(1)))


ENC_SHAPES = {0: (14, 28), 1: (14, 28), 2: (7, 7)}
# modality-specific auxiliary classification head for encoder-only loss (Eq. 17)
class AuxHead(nn.Module):
    def __init__(self, n_class=N_CLASS):
        super().__init__()
        self.fc = nn.Linear(EMB_DIM, n_class)

    def forward(self, z):
        return self.fc(z)


class Fusion(nn.Module):
    """Local fusion module: mean of available embeddings -> classifier."""

    def __init__(self, n_class=N_CLASS):
        super().__init__()
        self.fc1 = nn.Linear(EMB_DIM, 64)
        self.fc2 = nn.Linear(64, n_class)

    def forward(self, embs):
        z = torch.stack(embs, 0).mean(0)
        return self.fc2(F.relu(self.fc1(z)))


def new_encoder(r):
    return Encoder(ENC_SHAPES[r])


# dataset specs: views fn, encoder factory, number of classes
SPECS = {
    "fmnist": {"views": modality_views, "enc": lambda r: Encoder(ENC_SHAPES[r]),
               "n_class": 10},
    "har": {"views": har_views, "enc": lambda r: Encoder1D(), "n_class": 6},
}


def get_vec(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def set_vec(model, vec):
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


# ---------------------------------------------------------------- vehicle
class Vehicle:
    def __init__(self, vid, idx, mods, mod_frac, quality, xtr, ytr, seed,
                 spec=None):
        self.spec = spec or SPECS["fmnist"]
        self.id = vid
        self.mods = mods
        self.quality = quality
        rng = np.random.RandomState(seed * 977 + vid)
        self.data = {}
        self.D = {}
        idx = np.array(idx)
        rng.shuffle(idx)  # random order; per-modality subsets are prefixes
        x = xtr[idx]
        y = ytr[idx]
        views = self.spec["views"](x)
        for r in mods:
            n_keep = min(max(int(len(idx) * mod_frac[r]), 20), len(idx))
            xv = views[r][:n_keep].clone()
            if quality[r] < 0.7:  # degraded sensing -> additive noise
                xv += torch.randn_like(xv) * (0.8 * (1 - quality[r]))
            self.data[r] = (xv, y[:n_keep].clone())
            self.D[r] = n_keep
        # joint samples shared by all modalities (prefix intersection)
        self.n_joint = min(self.D.values())
        self.enc = {r: self.spec["enc"](r).to(DEVICE) for r in mods}
        self.aux = {r: AuxHead(self.spec["n_class"]).to(DEVICE) for r in mods}
        self.fusion = Fusion(self.spec["n_class"]).to(DEVICE)
        lr = 0.02
        self.opt = {
            r: torch.optim.SGD(
                list(self.enc[r].parameters()) + list(self.aux[r].parameters()),
                lr=lr, momentum=0.9,
            )
            for r in mods
        }
        self.optf = torch.optim.SGD(self.fusion.parameters(), lr=lr,
                                    momentum=0.9)

    def local_train(self, steps=10, bs=32):
        """Eq. (17)-(19): encoder + aux loss per modality, then fusion."""
        for r in self.mods:
            xv, yv = self.data[r]
            opt = self.opt[r]
            for _ in range(steps):
                sel = torch.randint(0, len(xv), (min(bs, len(xv)),))
                xb, yb = xv[sel].to(DEVICE), yv[sel].to(DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(self.aux[r](self.enc[r](xb)), yb)
                loss.backward()
                opt.step()
        # fusion on the joint samples shared by all local modalities
        r0 = self.mods[0]
        _, yv0 = self.data[r0]
        optf = self.optf
        for _ in range(steps):
            sel = torch.randint(0, self.n_joint, (min(bs, self.n_joint),))
            yb = yv0[sel].to(DEVICE)
            embs = []
            for r in self.mods:
                xvr, _ = self.data[r]
                with torch.no_grad():
                    embs.append(self.enc[r](xvr[sel].to(DEVICE)))
            optf.zero_grad()
            loss = F.cross_entropy(self.fusion(embs), yb)
            loss.backward()
            optf.step()

    def aggregate(self, r, received, phi_agg=0.0):
        """Eq. (20): data-size weighted aggregation of own + received encoders.

        received: list of (vec, D_m, Q_m, staleness) tuples. With
        phi_agg > 0 the weight is staleness-discounted:
        D_m * Q_m * exp(-phi_agg * Delta).
        """
        if r not in self.enc or not received:
            return
        wn = float(self.D[r])
        acc = get_vec(self.enc[r]) * wn
        for vec, Dm, Qm, dl in received:
            w = float(Dm) * float(Qm) * float(np.exp(-phi_agg * dl))
            acc += vec.to(DEVICE) * w
            wn += w
        set_vec(self.enc[r], acc / wn)

    @torch.no_grad()
    def evaluate(self, xte_views, yte):
        embs = [self.enc[r](xte_views[r].to(DEVICE)) for r in self.mods]
        pred = self.fusion(embs).argmax(1)
        return (pred == yte.to(DEVICE)).float().mean().item()
