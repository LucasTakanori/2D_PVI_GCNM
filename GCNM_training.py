import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

import time
import copy
import scipy.io
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol

from pyeit.eit.fem import EITForward
from pyeit.eit.base import EitBase
from pyeit.mesh.shape import circle, thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle, PyEITAnomaly_Ellipse
from pyeit.mesh import set_perm, plot as mesh_plot

torch.cuda.is_available()
torch.cuda.get_device_name(0)

############################
## training sample 생성함수 ##
############################

def sample_point_in_fd(fd, bbox, rng, max_reject=2000):
    """
    fd(pts)->signed distance (<0: inside)를 갖는 임의 도메인에서
    bbox 내 재시도로 내부 점 1개를 샘플링.
    """
    (xmin, ymin), (xmax, ymax) = bbox
    for _ in range(max_reject):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if fd(np.array([[x, y]], dtype=float)) < 0:
            return np.array([x, y], dtype=float)
    return None  # 실패

def ellipse_boundary_points(a, b, theta, n=32):
    """
    중심 (0,0)인 타원 경계점들 생성 (회전 포함), shape: (n,2)
    """
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    # 축 기준 좌표
    pts = np.stack([a*np.cos(t), b*np.sin(t)], axis=1)  # (n,2)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    return pts @ R.T  # (n,2)

def ellipse_inside_fd(center, a, b, theta, fd, margin=0.0, n_check=48):
    """
    타원 경계의 여러 점을 검사하여 fd< -margin이면 '전부 내부'로 판정.
    margin>0이면 경계에서 더 안쪽만 허용(보수적).
    """
    bd = ellipse_boundary_points(a, b, theta, n=n_check) + center[None, :]
    vals = fd(bd.astype(float))
    return np.all(vals < -margin)

def generate_anomalies(
    n_inclusions,
    low_range,
    high_range,
    max_attempts=150,
    *,
    R=0.5,                 # 도메인 원판 반지름 (중심 (0,0))
    r_range=(0.1, 0.3),    # 포함체 반지름 범위
    gap=0.05,               # 포함체 간 안전 거리(비중첩 마진)
    p_high=0.5,            # high_range에서 뽑을 확률
    rng=None               # int seed 또는 np.random.Generator
):
    """
    원판(반지름 R) 내부에서 면적 균등하게 중심을 샘플링하여
    겹치지 않는 원형 anomaly들을 생성합니다.
    - 중심 (x,y): ρ = (R - r)*sqrt(U), θ ~ U(0, 2π)
    - 경계 밖으로 나가지 않도록 ρ 최대치를 (R - r)로 제한
    - 상호 간 거리 >= r_i + r_j + gap 보장
    """
    g = np.random.default_rng(rng)
    anomalies = []

    for _ in range(n_inclusions):
        placed = False
        for _ in range(max_attempts):
            # 1) 포함체 반지름
            r = g.uniform(*r_range)
            if (R - r) <= 0:
                # 포함체가 도메인보다 커서 배치 불가
                break

            # 2) 중심 (면적 균등 샘플링)
            rho   = (R - r) * np.sqrt(g.random())
            theta = 2 * np.pi * g.random()
            x = rho * np.cos(theta)
            y = rho * np.sin(theta)
            center = np.array([x, y], dtype=float)

            # 3) 물성
            perm = g.uniform(*high_range) if (g.random() < p_high) else g.uniform(*low_range)

            # 4) 비중첩 검사
            ok = True
            for a in anomalies:
                dist = np.linalg.norm(center - np.array(a["center"]))
                if dist < (r + a["r"] + gap):
                    ok = False
                    break
            if not ok:
                continue

            # 5) 배치 확정
            anomalies.append({"center": center, "r": r, "perm": perm})
            placed = True
            break

        if not placed:
            print("⚠️ Warning: failed to place non-overlapping anomaly after max_attempts")

    return [PyEITAnomaly_Circle(center=a["center"], r=a["r"], perm=a["perm"]) for a in anomalies]

from pyeit.mesh.wrapper import PyEITAnomaly_Ellipse

def generate_anomalies_ellipse_ordered_poly(
    n_inclusions,
    low_range, high_range,
    *,
    fd,                      # 예: shape.thorax
    bbox,                    # [[xmin,ymin],[xmax,ymax]]  (도메인 외접 bbox)
    a_range=(0.10, 0.30),
    b_range=(0.10, 0.30),
    theta_range=(0.0, 2*np.pi),
    gap=0.10,                # 포함체 간 간격(외접원 기준)
    margin=0.00,             # 경계에서 더 안쪽만 허용하고 싶을 때 >0
    p_high=0.5,
    max_attempt=5000,
    rng=None,
    boundary_checks=48,      # 타원 경계 샘플링 점 수
):
    """
    thorax 같은 다각형 도메인에 대해 요구사항 1~7 구현:
      - 축 먼저 전부 생성 → r_eff = max(a,b) 큰 것부터 배치
      - 중심은 도메인 내부에서 샘플링(rejection)
      - 타원 경계를 여러 점 검사하여 '도메인 밖 아님' 보증
      - 기존 배치와는 외접원 + gap으로 비중첩 검사
      - 실패 시 '남은 것만' 축 재생성해서 반복
    """
    g = np.random.default_rng(rng)
    placed = []  # dict(center,a,b,theta,perm,r_eff)

    def sample_axes(k):
        A = g.uniform(*a_range, size=k)
        B = g.uniform(*b_range, size=k)
        a = np.maximum(A, B)
        b = np.minimum(A, B)
        r_eff = a  # 외접원 반경
        order = np.argsort(-r_eff)  # 큰 것부터
        return list(zip(a[order], b[order]))

    def sample_theta(): return g.uniform(*theta_range)
    def sample_perm():  return g.uniform(*high_range) if (g.random() < p_high) else g.uniform(*low_range)

    def non_overlap(center, r_eff):
        for an in placed:
            if np.linalg.norm(center - an["center"]) < (r_eff + an["r_eff"] + gap):
                return False
        return True

    remaining = n_inclusions
    while remaining > 0:
        axes_list = sample_axes(remaining)
        newly_placed = 0

        for (a, b) in axes_list:
            r_eff = a
            success = False
            for _ in range(max_attempt):
                # 1) 중심: 도메인 내부에서 재시도
                center = sample_point_in_fd(fd, bbox, g, max_reject=2000)
                if center is None:
                    continue
                theta = sample_theta()

                # 2) 도메인 내부성(타원 경계 샘플링) 체크
                if not ellipse_inside_fd(center, a, b, theta, fd, margin=margin, n_check=boundary_checks):
                    continue

                # 3) 비중첩 검사(빠른 외접원 + gap)
                if not non_overlap(center, r_eff):
                    continue

                # 4) 배치 확정
                placed.append({
                    "center": center,
                    "a": float(a), "b": float(b),
                    "theta": float(theta),
                    "perm": float(sample_perm()),
                    "r_eff": float(r_eff),
                })
                newly_placed += 1
                success = True
                break

            if not success:
                # 이번 라운드에 남은 것들 전부 실패 → 요구 6: '이미 배치된 것' 외 나머지 초기화
                break

        # 남은 개수 업데이트
        remaining = n_inclusions - len(placed)
        # 요구 7: 남은 개수만큼 다시 2번(축 재생성)부터 반복

        # 안전장치: 너무 빡빡하면 margin↓, gap↓, a/b 상한↓를 고려
        if newly_placed == 0 and remaining > 0 and max_attempt <= 1:
            pass

    anomalies = [
        PyEITAnomaly_Ellipse(
            center=an["center"], a=an["a"], b=an["b"],
            theta=an["theta"], degree=False, perm=an["perm"]
        )
        for an in placed
    ]

    meta = {
        "centers": np.array([an["center"] for an in placed], dtype=float),
        "a": np.array([an["a"] for an in placed], dtype=float),
        "b": np.array([an["b"] for an in placed], dtype=float),
        "theta": np.array([an["theta"] for an in placed], dtype=float),
        "perm": np.array([an["perm"] for an in placed], dtype=float),
        "r_eff": np.array([an["r_eff"] for an in placed], dtype=float),
    }
    return anomalies, meta


def save_sample_npz(out_dir, idx, bkg, meta, sigma_vec, V_vec):
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"sample_{idx:05d}.npz")
    np.savez_compressed(
        fn,
        bkg=bkg,
        centers=meta["centers"],
        a=meta["a"], b=meta["b"],
        theta=meta["theta"],
        perm=meta["perm"],
        r_eff=meta["r_eff"],
        sigma=sigma_vec,
        V=V_vec
    )
    return fn

def save_sample_plot(plots_dir, idx, mesh_obj, sigma_vec, vmin=0.15, vmax=0.95):
    os.makedirs(plots_dir, exist_ok=True)
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:,0], pts[:,1]
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    im = ax.tripcolor(x, y, tri, sigma_vec, shading="flat", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("σ")
    fig.suptitle(f"Sample #{idx:05d}", y=0.98)
    out = os.path.join(plots_dir, f"sample_{idx:05d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


##########################
## training sample 생성 ##
##########################

# Mesh 조건
n_el = 32  # number of electrodes
n_samples = 500  # number of samples
h0_fwd = 0.045
h0_inv = 0.05


# Conductivity value distribution (Training)
n_inclusions_range = [1, 4]
bkg_range = [0.40, 0.43]
low_range = [0.15, 0.25]
high_range = [0.65, 0.95]

use_thorax = False
if use_thorax:
    out_root = "samples_thorax"
    mesh_forward = mesh.create(n_el, h0=h0_fwd, fd=thorax)
    mesh_inverse = mesh.create(n_el, h0=h0_inv, fd=thorax)
    # bbox는 메쉬로부터 얻기
    xmin, ymin = mesh_inverse.node[:, :2].min(axis=0)
    xmax, ymax = mesh_inverse.node[:, :2].max(axis=0)
    bbox_dom = np.array([[xmin, ymin], [xmax, ymax]], dtype=float)
    fd_dom = thorax
else:
    out_root = "samples_circle"
    mesh_forward = mesh.create(n_el, h0=h0_fwd, fd=circle)
    mesh_inverse = mesh.create(n_el, h0=h0_inv, fd=circle)
    # 원형이라면 기존 방식(R) 사용 가능하지만, 통일성을 위해 fd/bbox도 쓸 수 있음
    xmin, ymin = -1.0, -1.0
    xmax, ymax =  1.0,  1.0
    bbox_dom = np.array([[xmin, ymin], [xmax, ymax]], dtype=float)
    fd_dom = circle

# 프로토콜 생성
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

mesh_forward.print_stats()
mesh_inverse.print_stats()

# ==========================
# 샘플 생성 및 저장 (요구 1~7)
# ==========================

# 저장 디렉토리
npz_dir   = os.path.join(out_root, "npz")
plots_dir = os.path.join(out_root, "plots")
os.makedirs(npz_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

sigma_list = []
V_list = []

noise_scale = 0.005

for i in range(n_samples):
    # 1) 이번 샘플의 anomaly 개수(샘플마다 정해짐: 고정값이나 랜덤값 둘 다 OK)
    n_inc = np.random.randint(n_inclusions_range[0], n_inclusions_range[1] + 1)

    # 배경도 샘플마다
    bkg = np.random.uniform(*bkg_range)

    # 2~7) 생성(가장 큰 것부터 배치, 실패 시 남은 것 리샘플) + 저장
    anomalies, meta = generate_anomalies_ellipse_ordered_poly(
        n_inclusions=n_inc,
        low_range=low_range, high_range=high_range,
        fd=fd_dom,
        bbox=bbox_dom,
        a_range=(0.15, 0.30),
        b_range=(0.15, 0.30),
        theta_range=(0.0, 2 * np.pi),
        gap=0.10,
        margin=0.05,  # 경계에서 더 안쪽만 허용하고 싶으면 0.02~0.05 등으로 설정
        p_high=0.5,
        max_attempt=5000,
        rng=None,
        boundary_checks=128  # thorax는 굴곡이 있으니 샘플 점 수를 약간 늘리면 안전
    )

    # 메쉬에 anomaly 적용
    mesh_fwd = mesh.set_perm(mesh_forward, anomaly=anomalies, background=bkg)
    mesh_inv = mesh.set_perm(mesh_inverse, anomaly=anomalies, background=bkg)

    # forward solve
    fwd = EITForward(mesh_fwd, protocol_obj)
    vp = fwd.solve_eit(perm=mesh_fwd.perm)
    V = vp + noise_scale * np.mean(np.abs(vp)) * np.random.randn(*vp.shape)

    # 누적(학습 파이프라인용)
    sigma_list.append(mesh_inv.perm.copy())
    V_list.append(V.copy())

    # 저장
    npz_path = save_sample_npz(npz_dir, i, bkg, meta, mesh_inv.perm.copy(), V.copy())
    png_path = save_sample_plot(plots_dir, i, mesh_inverse, mesh_inv.perm.copy(),
                                vmin=low_range[0], vmax=high_range[1])

    print(f"Sample {i+1}/{n_samples} done.  npz: {npz_path}  plot: {png_path}")

# 배열로 변환(기존 파이프라인 계속 사용)
sigma_array = np.stack(sigma_list)
V_array     = np.stack(V_list)

print(sigma_array.shape, V_array.shape)
print("✅ All training samples generated & saved.")

# generate mesh edge_index
elements = mesh_inverse.element
num_elements = elements.shape[0]
A = sp.lil_matrix((num_elements, num_elements), dtype=int)
for i, elem_i in enumerate(elements):
    for j, elem_j in enumerate(elements):
        if i != j and len(set(elem_i) & set(elem_j)) > 0:
            A[i, j] = 1

adj = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

############################
## training sample 시각화 ##
############################

# plot N samples
n_show = 10
total_samples = sigma_array.shape[0]

# 무작위 인덱스 선택
random_indices = np.random.choice(total_samples, n_show, replace=False)

# mesh info
pts = mesh_inverse.node
tri = mesh_inverse.element
x, y = pts[:, 0], pts[:, 1]

# conductivity range
vmin = 0.15
vmax = 0.95

# plot
fig, axs = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
for i, idx in enumerate(random_indices):
    sigma_sample = sigma_array[idx]
    im = axs[i].tripcolor(
        x, y, tri, sigma_sample,
        shading="flat", cmap="viridis",
        vmin=vmin, vmax=vmax
    )
    axs[i].set_title(f"Sample #{idx}")
    axs[i].set_aspect("equal")
    axs[i].axis("off")

# colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.suptitle("Randomly Selected Conductivity Samples", fontsize=14)
plt.show()


################################################

def initializeDataset(sigma_array, V_array, adj):

    N_samples = sigma_array.shape[0]
    N_elements = sigma_array.shape[1]

    # The truth and inputs
    data_true = torch.tensor(sigma_array)
    data_in = 1.0 * torch.ones((N_samples, N_elements), dtype=torch.float64)
    data_V = torch.tensor(V_array)
    # Now create a list of data objects
    dataset = []
    for i in range(N_samples):
        dataset.append(Data(edge_index=adj,
                            x=data_in[i, :].unsqueeze(dim=1),
                            y=data_true[i, :].unsqueeze(dim=1),
                            V=data_V[i, :].unsqueeze(dim=1)))

    # Return the dataset
    return dataset


# print(V_array[0])
# print(V_array.shape)
#
# plt.plot(V_array[0])
# plt.show()


def computeLMUpdates(dataset, mesh_template, protocol, lambda_LM=0.1):
    for i in range(len(dataset)):
        data = dataset[i]
        V = data.V
        sigma_current = data.x[:, 0].unsqueeze(1)  # [n_elements, 1]

        # 1. 해당 σ를 mesh에 설정
        mesh_template.perm = sigma_current.squeeze().cpu().numpy()

        # 2. 샘플마다 fwd 생성
        fwd = EITForward(mesh_template, protocol)

        # 3. Jacobian 계산
        jac, vp = fwd.compute_jac(mesh_template.perm, normalize=True)
        J = torch.tensor(jac, device=V.device)
        U = torch.tensor(vp, device=V.device).unsqueeze(dim=1)

        # 4. 전압 노이즈 추가 및 δσ 계산
        residual = U - V
        JTJ = J.T @ J
        I = torch.eye(JTJ.size(0), device=V.device)
        delta_sigma = - torch.linalg.solve(JTJ + lambda_LM * I, J.T @ residual)

        # 5. feature = [σ, δσ]
        H = torch.cat((sigma_current.to(V.device), delta_sigma.detach()), dim=1)
        dataset[i].x = H

    return dataset


class GCNBlock(torch.nn.Module):
    # This class is for the GNN model

    def __init__(self, channels):
        super(GCNBlock, self).__init__()
        self.channels = channels

        # Make a list of the graph convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        in_channels = 2
        for out_channels in channels:
            self.conv_layers.append(GCNConv(in_channels, out_channels))
            in_channels = out_channels
        self.conv_layers.append(GCNConv(in_channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # A function for resetting the parameters
        for layer in self.conv_layers:
            layer.reset_parameters()

    def forward(self, data):
        # The forward pass
        x, ei = data.x, data.edge_index
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, ei)
            if i < (len(self.conv_layers) - 1):  # No relu on final layer
                x = torch.nn.functional.relu(x)
        return x


def trainModel(model, dataset, optimizer, split, batch_size, max_epochs, patience, start_time):
    # This is the main training function

    # Split the dataset into training and validation and create dataLoaders
    split = int(split * len(dataset))
    dataset_tr = dataset[:split]
    dataset_va = dataset[split:]
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=False)

    # Prepare some things
    loss_tr = torch.zeros((max_epochs))
    loss_va = torch.zeros((max_epochs))
    pat = patience
    loss_va_min = 10e6

    # Start main training loop
    for epoch in range(max_epochs):

        # Do training and validation
        loss_tr[epoch] = train(model, loader_tr, optimizer, len(dataset_tr))
        loss_va[epoch] = valid(model, loader_va, len(dataset_va))

        # Print an update every once in a while
        if (epoch) % 1 == 0:
            elapsed = time.time() - start_time
            hrs = int(elapsed // 3600)
            mins = int((elapsed - 3600 * hrs) // 60)
            sec = int((elapsed - 3600 * hrs - 60 * mins) // 1)
            print('# ({:02d}:{:02d}:{:02d}) Epoch {:3d} | Training Loss {:.4f} | Validation Loss {:.4f}'
                  .format(hrs, mins, sec, epoch, loss_tr[epoch] * 1000, loss_va[epoch] * 1000))

        # Maybe stop early due to non-decreasing loss_va
        if loss_va[epoch] <= loss_va_min:
            loss_va_min = loss_va[epoch]
            best_model = copy.deepcopy(model)
            pat = patience
            best_epoch = epoch
        else:
            pat -= 1
            if pat == 0:
                for model_w, best_w in zip(model.parameters(), best_model.parameters()):
                    model_w.data = best_w.data
                print("# Training stopped early on epoch", best_epoch, "and best weights loaded back...")
                print("#=========================================================")
                break

    return model, loss_tr, loss_va

def train(model, loader_tr, optimizer, n):
    model.train() # Puts the model in training mode (ex. dropout will happen)

    loss_all = 0
    for data in loader_tr:
        optimizer.zero_grad()
        sigma_k = model(data)
        sigma_true = data.y
        loss = F.mse_loss(sigma_k, sigma_true)
        loss.backward()
        loss_all += data.num_graphs*loss.item()
        optimizer.step()
    return loss_all / n

def valid(model, loader_va, n):
    model.eval() # Puts the model in eval mode (ex. dropout will NOT happen)

    loss_all = 0
    with torch.no_grad():
        for data in loader_va:
            sigma_pred = model(data)
            sigma_true = data.y
            loss = F.mse_loss(sigma_pred, sigma_true)
            loss_all += data.num_graphs*loss.item()
    return loss_all / n

def applyModel(model, dataset):
    # Apply the model to the dataset
    model.eval()

    # Initialize storage
    predictions = torch.zeros((len(dataset), dataset[0].x.shape[0]), device=dataset[0].x.device)

    with torch.no_grad():
        for i in range(len(dataset)):
            predictions[i,:] = model(dataset[i]).squeeze()
            dataset[i].x = predictions[i,:].unsqueeze(dim=1)

    return dataset, predictions.to('cpu')

# Set a default tensor dtype
# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
torch.set_default_dtype(torch.float64)

# Where are we doing the training?
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("Using device:", device)

# How many iterations of GCNM
iterations = 10

# What does the graph structure look like?
N_samples = n_samples
N_nodes = mesh_inverse.element.shape[0]

# What do the models look like?
channels = [250,250,250]

# Set hyperparameters for training and model
model_name = 'sample_models'
split = 0.8
learning_rate = 0.002
batch_size = 10
max_epochs = 10000
patience = 200



#===========================#
# Prepare for the main loop #
#===========================#

# Set up storage tensors
TRUTHS      = torch.zeros((              N_samples, N_nodes))
PREDICTIONS = torch.zeros((1+iterations, N_samples, N_nodes))
UPDATES     = torch.zeros((  iterations, N_samples, N_nodes))
LOSS_TR     = torch.zeros((  iterations, max_epochs))
LOSS_VA     = torch.zeros((  iterations, max_epochs))

# Initialize a random dataset
dataset = initializeDataset(sigma_array, V_array, adj)    #<---- Load in a real dataset here instead of using random numbers
for i in range(N_samples):
    TRUTHS[i,:] = dataset[i].y.squeeze()
    PREDICTIONS[0,i,:] = dataset[i].x.squeeze()



#=================#
# Start main loop #
#=================#

start_time = time.time()
for k in range(iterations):
    print("Starting iteration", str(k))

    # LM Updates for each sample in the dataset
    dataset = computeLMUpdates(dataset, mesh_inverse, protocol_obj)
    for i in range(N_samples):
        UPDATES[k,i,:] = dataset[i].x[:,1]

    # Move the dataset to the device
    for i in range(len(dataset)):
        dataset[i].x = dataset[i].x.to(device)
        dataset[i].y = dataset[i].y.to(device)
        dataset[i].edge_index = dataset[i].edge_index.to(device)

    # Initialize a model and move to device
    model = GCNBlock(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model using the dataset
    model, LOSS_TR[k,:], LOSS_VA[k,:] = trainModel(model, dataset, optimizer, split, batch_size, max_epochs, patience, start_time)

    # Apply the model to the whole dataset
    dataset, PREDICTIONS[k+1,:,:] = applyModel(model, dataset)

    # Move the model and data back to cpu from device
    model = model.to('cpu')
    for i in range(len(dataset)):
        dataset[i].x = dataset[i].x.to('cpu')
        dataset[i].y = dataset[i].y.to('cpu')
        dataset[i].edge_index = dataset[i].edge_index.to('cpu')

    # Finally, save the model
    save_name = 'models/' + model_name + '_' + str(k) + '.pt'
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    torch.save(model.state_dict(), save_name)
    print("Saved model as:", save_name)
total_time = time.time() - start_time



#================#
# Save some info #
#================#

save_name = 'data/' + model_name + '_training_output.mat'
# Create the 'data' directory if it doesn't exist
os.makedirs(os.path.dirname(save_name), exist_ok=True)
save_data = {
    'model_name'  : model_name,
    'iterations'  : iterations,
    'channels'    : np.array(channels),
    'TRUTHS'      : TRUTHS.numpy(),
    'PREDICTIONS' : PREDICTIONS.numpy(),
    'UPDATES'     : UPDATES.numpy(),
    'LOSS_TR'     : LOSS_TR.numpy(),
    'LOSS_VA'     : LOSS_VA.numpy(),
    'total_time'  : total_time
}
scipy.io.savemat(save_name, save_data)