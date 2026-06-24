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
import os
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


# ---------------------------------------------------------------- MM-Fi
# Action recognition (27 classes) from three modalities used in the paper:
#   r=0 skeleton (3D body keypoints), r=1 mmWave point cloud (BEV voxels),
#   r=2 depth image. Preprocessed into per-window samples by sim/prep_mmfi.py.
MMFI_KEYS = {0: "xsk", 1: "xmm", 2: "xdp"}
MMFI_N_CLASS = 27


class MultiModalArray:
    """Indexable container of per-modality arrays sharing a sample axis.

    Lets the FL pipeline treat heterogeneous-shaped modalities like a single
    dataset: arr[idx] returns {r: tensor[idx]} so spec['views'] is a pass-through.
    """

    def __init__(self, mods):
        self.mods = mods  # {r: np.ndarray [N, ...]}
        self._n = len(next(iter(mods.values())))

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {r: torch.as_tensor(arr[idx]) for r, arr in self.mods.items()}


def mmfi_views(x):
    """x is already the {r: tensor} dict from MultiModalArray.__getitem__."""
    return x


def load_mmfi(cache="results/data/mmfi_cache.npz"):
    """Load preprocessed MM-Fi cache -> (train container, ytr, subj_tr,
    test container, yte). Test split = subjects with id % 5 == 0 held out."""
    d = np.load(cache)
    present = [r for r, k in MMFI_KEYS.items() if k in d]
    y = torch.tensor(d["y"]).long()
    subj = d["subj"]
    mods = {r: d[MMFI_KEYS[r]] for r in present}
    te_mask = (subj % 5 == 0)
    tr_mask = ~te_mask
    tr = MultiModalArray({r: mods[r][tr_mask] for r in present})
    te = MultiModalArray({r: mods[r][te_mask] for r in present})
    return tr, y[tr_mask], subj[tr_mask], te, y[te_mask], present


def partition_mmfi(subj_tr, n_veh, seed):
    """Subject-based non-IID partition (same idea as UCI HAR)."""
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


class SkeletonEnc(nn.Module):
    """1D-CNN over skeleton channels-over-time [B,51,T]."""

    def __init__(self, in_ch=51):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 4, EMB_DIM)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        return F.relu(self.fc(x.flatten(1)))


class MmwaveEnc(nn.Module):
    """2D-CNN over mmWave BEV voxel grid [B,2,32,32]."""

    def __init__(self, in_ch=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, EMB_DIM)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return F.relu(self.fc(x.flatten(1)))


class DepthEnc(nn.Module):
    """2D-CNN over depth image [B,1,112,112]."""

    def __init__(self, in_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 14 * 14, EMB_DIM)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return F.relu(self.fc(x.flatten(1)))


def _mmfi_enc(r):
    return {0: SkeletonEnc, 1: MmwaveEnc, 2: DepthEnc}[r]()


# ------------------------------------------------------------- Raymobtime
# Vehicular multimodal mmWave BEAM SELECTION (the FLASH/ITU-ML5G task): predict
# the best Tx-Rx beam pair from three vehicle sensors -- GPS coordinate, camera
# image, LiDAR occupancy cube. This is the standard vehicular multimodal-FL
# benchmark (Klautau et al.; Salehi et al. FLASH; Imperial federated beam sel.).
RAYMOB_KEYS = {0: "coord", 1: "image", 2: "lidar"}


def load_raymob(root="results/data/raymobtime/bs_baseline_data",
                n_class=256, coarse=False):
    """Load Raymobtime baseline npz -> (train container, ytr, region_tr,
    test container, yte, present). Modalities pre-permuted to channel-first.

    Label = best beam pair (argmax |gain| over the 8x32 codebook = 256 classes);
    coarse=True collapses to the 8 Tx sectors. region = spatial bin of the GPS
    coordinate, used for location-correlated non-IID partition across vehicles.
    """
    coord = np.load(os.path.join(root, "coord_input/coord_input.npz"))["coordinates"].astype(np.float32)
    img = np.load(os.path.join(root, "image_input/img_input_20.npz"))["inputs"].astype(np.float32) / 255.0
    lid = np.load(os.path.join(root, "lidar_input/lidar_input.npz"))["input"].astype(np.float32)
    beam = np.load(os.path.join(root, "beam_output/beams_output.npz"))["output_classification"]
    power = np.abs(beam).reshape(len(beam), -1)        # (N, 256)
    y = (power.reshape(len(beam), 8, 32).max(2).argmax(1) if coarse
         else power.argmax(1)).astype(np.int64)        # 8 or 256 classes
    # channel-first layouts for torch convs
    img = np.transpose(img, (0, 3, 1, 2))              # (N,1,48,81)
    lid = np.transpose(lid, (0, 3, 1, 2))              # (N,10,20,200)
    cmu, csd = coord.mean(0), coord.std(0) + 1e-6
    coord = (coord - cmu) / csd
    N = len(y)
    # location-correlated regions (spatial bins on normalized coordinate)
    region = (np.clip(((coord[:, 0] - coord[:, 0].min()) /
                       (np.ptp(coord[:, 0]) + 1e-9) * 50).astype(int), 0, 49))
    rng = np.random.RandomState(0)
    te_mask = rng.rand(N) < 0.2
    tr_mask = ~te_mask
    mods = {0: coord, 1: img, 2: lid}
    tr = MultiModalArray({r: mods[r][tr_mask] for r in mods})
    te = MultiModalArray({r: mods[r][te_mask] for r in mods})
    return (tr, torch.tensor(y[tr_mask]), region[tr_mask],
            te, torch.tensor(y[te_mask]), [0, 1, 2])


def partition_raymob(region_tr, n_veh, seed):
    """Location-correlated non-IID: contiguous spatial regions -> vehicles."""
    rng = np.random.RandomState(seed)
    order = np.argsort(region_tr, kind="stable")
    shards = np.array_split(order, n_veh)
    idx = list(range(n_veh))
    rng.shuffle(idx)
    return [shards[i] for i in idx]


class CoordEnc(nn.Module):
    """MLP over the 2-D GPS coordinate."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, EMB_DIM)

    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class ImageEnc(nn.Module):
    """2D-CNN over the 48x81 grayscale image."""

    def __init__(self, in_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 12 * 20, EMB_DIM)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return F.relu(self.fc(x.flatten(1)))


class LidarEnc(nn.Module):
    """2D-CNN over the 10x20x200 LiDAR occupancy cube (10 channels)."""

    def __init__(self, in_ch=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((5, 25))
        self.fc = nn.Linear(32 * 5 * 25, EMB_DIM)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return F.relu(self.fc(x.flatten(1)))


def _raymob_enc(r):
    return {0: CoordEnc, 1: ImageEnc, 2: LidarEnc}[r]()


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
# multiplier on the additive sensor noise for low-quality modalities; engine may
# override per run (harsh-quality regime cranks it so the encoder genuinely fails)
QUAL_NOISE = 0.8


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
              mean_size=420, p_mod=0.72, q_low_frac=0.35, starve_frac=0.0,
              starve_ids=None, starve_mod=None, q_low_range=(0.25, 0.55)):
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

    mods, mod_frac, quality = assign_heterogeneity(
        n_veh, rng, p_mod, q_low_frac, starve_frac,
        starve_ids=starve_ids, starve_mod=starve_mod, q_low_range=q_low_range)
    return veh_idx, mods, mod_frac, quality


# mod_frac below this sentinel marks a deliberately starved modality, whose
# local-data floor is relaxed (see Vehicle.__init__). Normal mod_frac is
# clipped to [0.05, 1.0], so default partitions never trigger it.
STARVE_FRAC_SENTINEL = 0.003


def assign_heterogeneity(n_veh, rng, p_mod=0.72, q_low_frac=0.35,
                         starve_frac=0.0, starve_ids=None, starve_mod=None,
                         universe=None, q_low_range=(0.25, 0.55)):
    """Random modality subsets, per-modality availability, sensing quality.

    If starve_frac > 0, that fraction of vehicles is made data-starved: one of
    their owned modalities is given near-zero data and low sensing quality, so
    a useful encoder for that modality can only come from V2V dissemination.
    `universe` restricts the modality set (e.g. only modalities present in the
    dataset cache); defaults to all N_MOD modalities.
    """
    univ = list(range(N_MOD)) if universe is None else list(universe)
    mods, mod_frac, quality = [], [], []
    for i in range(n_veh):
        m = [r for r in univ if rng.rand() < p_mod]
        if not m:
            m = [int(rng.choice(univ))]
        mods.append(sorted(m))
        # per-modality data availability (some vehicles have very little
        # data for one of their modalities -> high learning need)
        fr = {r: float(np.clip(rng.beta(2.0, 1.2), 0.05, 1.0)) for r in m}
        mod_frac.append(fr)
        # quality: a fraction of drives are "night/rain" with degraded sensing
        q = {}
        for r in m:
            q[r] = float(rng.uniform(*q_low_range)) if rng.rand() < q_low_frac \
                else float(rng.uniform(0.8, 1.0))
        quality.append(q)

    if starve_ids is not None:
        victims = list(starve_ids)           # location-correlated starvation
    else:
        n_starve = int(round(starve_frac * n_veh))
        victims = (rng.choice(n_veh, size=min(n_starve, n_veh), replace=False)
                   if n_starve > 0 else [])
    for i in victims:
        owned = mods[i]
        if starve_mod is not None:
            if starve_mod not in owned:
                continue                     # victim lacks the target modality
            r = starve_mod
        else:
            r = int(rng.choice(owned))       # starve one owned modality
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
    "mmfi": {"views": mmfi_views, "enc": _mmfi_enc, "n_class": MMFI_N_CLASS},
    "raymob": {"views": mmfi_views, "enc": _raymob_enc, "n_class": 256},
    "raymob8": {"views": mmfi_views, "enc": _raymob_enc, "n_class": 8},
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
                xv += torch.randn_like(xv) * (QUAL_NOISE * (1 - quality[r]))
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

    @torch.no_grad()
    def _feats(self, r, xview, bs=256):
        """Batched frozen-encoder features for modality r (N, EMB_DIM)."""
        out = []
        for s in range(0, len(xview), bs):
            out.append(self.enc[r](xview[s:s + bs].to(DEVICE)))
        return torch.cat(out)

    def encoder_probe_acc(self, xtr_v, ytr, xte_v, yte, epochs=120, bs=256):
        """Pure ENCODER quality per modality: freeze the encoder and fit a fresh
        linear probe on GLOBAL train features, then score on global test. This
        isolates the representation from the fusion model and from the vehicle's
        own (possibly 3-sample) aux head, so a useful encoder received over V2V
        shows up even when the fused accuracy does not."""
        ytr_d, yte_d = ytr.to(DEVICE), yte.to(DEVICE)
        out = {}
        for r in self.mods:
            ftr = self._feats(r, xtr_v[r], bs).detach()
            fte = self._feats(r, xte_v[r], bs).detach()
            probe = nn.Linear(EMB_DIM, self.spec["n_class"]).to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=0.01)
            for _ in range(epochs):
                sel = torch.randint(0, len(ftr), (min(512, len(ftr)),),
                                    device=DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(probe(ftr[sel]), ytr_d[sel])
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = (probe(fte).argmax(1) == yte_d).float().mean().item()
            out[r] = acc
        return out
