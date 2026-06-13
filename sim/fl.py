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
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
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
    """Split a batch of images [B,1,28,28] into the three modality views.

    Redundant views: each carries most of the class signal, so a single
    modality already classifies well (low complementarity).
    """
    top = x[:, :, :14, :]
    bot = x[:, :, 14:, :]
    radar = F.avg_pool2d(x, 4)  # [B,1,7,7]
    return {0: top, 1: bot, 2: radar}


# sensor-noise std for complementary modalities; engine may override per run
COMP_NOISE = 0.7


def complementary_views(x, noise=None):
    """Complementary modalities: three disjoint vertical thirds of the image,
    each kept at full [28,28] resolution (rest zeroed) plus independent sensor
    noise. A single modality sees only ~1/3 of the object under noise (weak),
    while fusing all three recovers the full object (strong). This makes the
    multimodal task genuinely sharing-dependent: a vehicle with a poor encoder
    for one modality benefits from receiving a better one.
    """
    noise = COMP_NOISE if noise is None else noise
    bounds = [(0, 9), (9, 19), (19, 28)]  # disjoint thirds
    out = {}
    for r, (a, b) in enumerate(bounds):
        v = torch.zeros_like(x)
        v[:, :, :, a:b] = x[:, :, :, a:b]
        v = v + torch.randn_like(v) * noise
        out[r] = v
    return out


def partition(y_train, n_veh, seed, dirichlet=0.5, size_lognorm=0.6,
              mean_size=420, p_mod=0.72, q_low_frac=0.35, starve_frac=0.0):
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
                                                   q_low_frac, starve_frac)
    return veh_idx, mods, mod_frac, quality


# mod_frac below this sentinel marks a deliberately starved modality, whose
# local-data floor is relaxed (see Vehicle.__init__). Normal mod_frac is
# clipped to [0.05, 1.0], so default partitions never trigger it.
STARVE_FRAC_SENTINEL = 0.003


def assign_heterogeneity(n_veh, rng, p_mod=0.72, q_low_frac=0.35,
                         starve_frac=0.0):
    """Random modality subsets, per-modality availability, sensing quality.

    If starve_frac > 0, that fraction of vehicles is made data-starved: one of
    their owned modalities is given near-zero data and low sensing quality, so
    a useful encoder for that modality can only come from V2V dissemination.
    """
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

    n_starve = int(round(starve_frac * n_veh))
    if n_starve > 0:
        victims = rng.choice(n_veh, size=min(n_starve, n_veh), replace=False)
        for i in victims:
            r = int(rng.choice(mods[i]))  # starve one owned modality
            mod_frac[i][r] = STARVE_FRAC_SENTINEL
            quality[i][r] = float(rng.uniform(0.25, 0.4))
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
    """Local fusion module. forward takes a dict {modality r: emb [B,EMB_DIM]}.

    mode='mean' (default): average available embeddings (legacy behaviour).
    mode='concat': fixed per-modality slots (zeros for absent modalities), so
    the classifier uses modalities distinctly and an improved/received encoder
    that fills a slot directly raises accuracy -- needed for sharing to help.
    """

    def __init__(self, n_class=N_CLASS, mode="mean", n_mod=N_MOD):
        super().__init__()
        self.mode = mode
        self.n_mod = n_mod
        in_dim = EMB_DIM * n_mod if mode == "concat" else EMB_DIM
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, n_class)

    def forward(self, emb_by_mod):
        any_emb = next(iter(emb_by_mod.values()))
        if self.mode == "concat":
            zero = torch.zeros(any_emb.shape[0], EMB_DIM, device=any_emb.device)
            z = torch.cat([emb_by_mod.get(r, zero)
                           for r in range(self.n_mod)], dim=1)
        else:
            z = torch.stack(list(emb_by_mod.values()), 0).mean(0)
        return self.fc2(F.relu(self.fc1(z)))


def new_encoder(r):
    return Encoder(ENC_SHAPES[r])


# dataset specs: views fn, encoder factory, number of classes
ENC_SHAPES_COMP = {0: (28, 28), 1: (28, 28), 2: (28, 28)}
SPECS = {
    "fmnist": {"views": modality_views, "enc": lambda r: Encoder(ENC_SHAPES[r]),
               "n_class": 10},
    "fmnist_comp": {"views": complementary_views,
                    "enc": lambda r: Encoder(ENC_SHAPES_COMP[r]),
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
                 spec=None, fusion_mode="mean"):
        self.spec = spec or SPECS["fmnist"]
        self.fusion_mode = fusion_mode
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
            # starved modalities (sentinel mod_frac) keep a much lower floor
            floor = 3 if mod_frac[r] < 0.02 else 20
            n_keep = min(max(int(len(idx) * mod_frac[r]), floor), len(idx))
            xv = views[r][:n_keep].clone()
            if quality[r] < 0.7:  # degraded sensing -> additive noise
                xv += torch.randn_like(xv) * (0.8 * (1 - quality[r]))
            self.data[r] = (xv, y[:n_keep].clone())
            self.D[r] = n_keep
        # joint samples shared by all modalities (prefix intersection)
        self.n_joint = min(self.D.values())
        self.enc = {r: self.spec["enc"](r).to(DEVICE) for r in mods}
        self.aux = {r: AuxHead(self.spec["n_class"]).to(DEVICE) for r in mods}
        self.fusion = Fusion(self.spec["n_class"], mode=fusion_mode).to(DEVICE)
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
            embs = {}
            for r in self.mods:
                xvr, _ = self.data[r]
                with torch.no_grad():
                    embs[r] = self.enc[r](xvr[sel].to(DEVICE))
            optf.zero_grad()
            loss = F.cross_entropy(self.fusion(embs), yb)
            loss.backward()
            optf.step()

    @torch.no_grad()
    def _mod_loss(self, r, cap=256):
        """Local modality-r loss f_{i,r} (Eq. 15) on this vehicle's own data."""
        xv, yv = self.data[r]
        n = min(cap, len(xv))
        xb, yb = xv[:n].to(DEVICE), yv[:n].to(DEVICE)
        return float(F.cross_entropy(self.aux[r](self.enc[r](xb)), yb))

    def aggregate(self, r, received, phi_agg=0.0, gated=False):
        """Eq. (20): data-size weighted aggregation of own + received encoders.

        received: list of (vec, D_m, Q_m, staleness) tuples. With
        phi_agg > 0 the weight is staleness-discounted:
        D_m * Q_m * exp(-phi_agg * Delta).

        If gated=True, the learning-gain guard of Eq. (learning_gain_lower_bound)
        is enforced: received encoders are folded in one at a time (in order of
        effective weight) and an encoder is accepted only if it reduces the
        local modality loss, i.e. G^learn = [.]^+ > 0. Encoders inconsistent
        with the receiver's local objective are rejected rather than averaged
        in, preventing negative transfer.
        """
        if r not in self.enc or not received:
            return

        if not gated:
            wn = float(self.D[r])
            acc = get_vec(self.enc[r]) * wn
            for vec, Dm, Qm, dl in received:
                w = float(Dm) * float(Qm) * float(np.exp(-phi_agg * dl))
                acc += vec.to(DEVICE) * w
                wn += w
            set_vec(self.enc[r], acc / wn)
            return

        base = get_vec(self.enc[r]).clone()
        f_cur = self._mod_loss(r)
        order = sorted(
            received,
            key=lambda t: -(float(t[1]) * float(t[2])
                            * float(np.exp(-phi_agg * t[3]))))
        wn = float(self.D[r])
        acc = base * wn
        accepted = False
        for vec, Dm, Qm, dl in order:
            w = float(Dm) * float(Qm) * float(np.exp(-phi_agg * dl))
            trial = (acc + vec.to(DEVICE) * w) / (wn + w)
            set_vec(self.enc[r], trial)
            f_try = self._mod_loss(r)
            if f_try < f_cur:          # G^learn > 0 -> accept
                acc = acc + vec.to(DEVICE) * w
                wn += w
                f_cur = f_try
                accepted = True
        # commit accepted aggregate (== base if nothing reduced the loss)
        set_vec(self.enc[r], acc / wn if accepted else base)

    @torch.no_grad()
    def evaluate(self, xte_views, yte):
        embs = {r: self.enc[r](xte_views[r].to(DEVICE)) for r in self.mods}
        pred = self.fusion(embs).argmax(1)
        return (pred == yte.to(DEVICE)).float().mean().item()
