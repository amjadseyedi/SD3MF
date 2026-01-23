import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def loadDataset(dataset):
    data = torch.load(f'{dataset}.pt')

    A_list = data['A_list']
    y = data['y']

    return A_list, y

# -------------------------------
# Dataset
# -------------------------------
class MultiViewMatrixDataset(Dataset):
    """Expects a list of views, each tensor of shape [N, d_v, d_v].
    Labels y: [N], floats for binary or ints for multiclass.
    """
    def __init__(self, A_list: List[torch.Tensor], y: torch.Tensor):
        assert isinstance(A_list, list) and len(A_list) >= 1, "A_list should be a non-empty list of tensors"
        n = A_list[0].shape[0]
        for A in A_list:
            assert A.shape[0] == n, "All views must have same number of samples"
            assert A.ndim == 3 and A.shape[1] == A.shape[2], "Each A^(v) must be [N, d_v, d_v]"
        self.A_list = A_list
        self.y = y
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx, [A[idx] for A in self.A_list], self.y[idx]


# -------------------------------
# Deep W module
# -------------------------------
class DeepW(nn.Module):
    """Deep nonnegative factorization: W = W1 W2 ... WL
    Shapes: [d, k1] * [k1, k2] * ... * [k_{L-1}, r] = [d, r]
    Nonnegativity via ReLU; row-normalize the final W.
    """
    def __init__(self, A, in_dim: int, out_dim: int, widths: List[int]):
        super().__init__()
        dims = [in_dim] + list(widths)
        self.factors = nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:]):
            P = torch.empty(a, b, device=A.device)
            try:
                if a >= b:
                    nn.init.orthogonal_(P)
                    P *= 0.001
                else:
                    nn.init.xavier_uniform_(P)
            except Exception:
                nn.init.xavier_uniform_(P)
            self.factors.append(nn.Parameter(P))

    def compose(self) -> torch.Tensor:
        W = None
        for P in self.factors:
            Pp = torch.relu(P)
            W = Pp if W is None else (W @ Pp)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-8)  # row-normalize
        return W

    def forward(self) -> torch.Tensor:
        return self.compose()

# -------------------------------
# Model
# -------------------------------
class MultiModalCollectiveMFDeep(nn.Module):
    """Deep multimodal collective MF:
      - For each view v, W^(v) = W1 ... WL (deep nonnegative).
      - Shared S_i across views (tensor [N, r, r]).
      - Learned simplex weights alpha^(v) via softmax over logits.
      - Classifier on z_i = sum_v alpha^(v) vec(W^(v)^T A_i^(v) W^(v)).
    """
    def __init__(
        self,
        A_list: List[torch.Tensor],
        n_samples: int,
        num_classes: int,
        widths: Optional[List[int]] = None,   # widths per layer (same for all views)
        layers: Optional[int] = None,         # if given, widths auto-taper when widths is None
    ):
        super().__init__()
        self.m = len(A_list)
        self.n = int(n_samples)
        self.num_classes = int(num_classes)

        # Build deep W per view
        self.deep_modules = nn.ModuleList()
        self._init_W_cache = []  # keep initial W for S init

        for v in range(self.m):
            d_v = A_list[v].shape[1]
            widths_v: List[int]
            if layers is not None and layers > 1:
                # auto widths taper if not provided
                if widths is None:
                    L = layers - 1  # number of hidden transitions before r
                    ws = []
                    last = d_v
                    for _ in range(L - 1):
                        last = max(widths_v[-1], int(round(last / 2)))
                        ws.append(last)
                    widths_v = ws
                else:
                    widths_v = list(widths)
            else:
                widths_v = []  # shallow -> one factor [d_v, r]

            mod = DeepW(A_list[v], d_v, widths_v[-1], widths_v)
            self.deep_modules.append(mod)
            with torch.no_grad():
                self._init_W_cache.append(mod.compose())

        # Initialize S from average M across views using the initial W
        with torch.no_grad():
            S0 = torch.zeros(self.n, widths_v[-1], widths_v[-1], dtype=A_list[0].dtype)
            for i in range(self.n):
                M_acc = 0.0
                for v in range(self.m):
                    Wv0 = self._init_W_cache[v]  # [d_v, r]
                    Aiv = A_list[v][i]           # [d_v, d_v]
                    Mv  = Wv0.T @ Aiv @ Wv0      # [r, r]
                    M_acc = M_acc + Mv
                S0[i] = (M_acc / self.m)
                S0[i] = 0.5 * (S0[i] + S0[i].T)
        self.S_params = nn.Parameter(S0)

        # alpha on simplex via softmax over logits
        self.alpha_logits = nn.Parameter(torch.zeros(self.m))

        # Linear classifier
        out_dim =  self.num_classes
        self.cls = nn.Linear(widths_v[-1] * widths_v[-1], out_dim, bias=True)

        nn.init.xavier_normal_(self.cls.weight, gain=0.001)  # smaller gain => smaller scale

        if self.cls.bias is not None:
            nn.init.constant_(self.cls.bias, 0.0)

    def get_alphas(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=0)

    def forward(
        self,
        idx: torch.Tensor,
        A_batch_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        B = idx.shape[0]

        S_batch = self.S_params[idx]  # [B, r, r]
        S_batch = 0.5 * (S_batch + S_batch.transpose(1, 2))

        alphas = self.get_alphas()  # [m]

        recon_list, M_list = [], []
        z = torch.zeros(B, S_batch.shape[1] * S_batch.shape[1], device=A_batch_list[0].device, dtype=A_batch_list[0].dtype)

        for v in range(self.m):
            A_v = A_batch_list[v]        # [B, d_v, d_v]
            Wv  = self.deep_modules[v]() # composed [d_v, r]

            # recon: W S_i W^T
            WS = torch.einsum('dr,brk->bdk', Wv, S_batch)             # [B, d_v, r]
            recon_v = torch.bmm(WS, Wv.t().unsqueeze(0).expand(B, -1, -1))  # [B, d_v, d_v]
            recon_list.append(recon_v)

            # M = W^T A W
            AW = torch.matmul(A_v, Wv)                 # [B, d_v, r]
            Mv = torch.matmul(AW.transpose(1, 2), Wv)  # [B, r, r]
            Mv = 0.5 * (Mv + Mv.transpose(1, 2))
            M_list.append(Mv)

            z = z + alphas[v] * Mv.reshape(B, -1)

        logits = self.cls(z)
        return recon_list, S_batch, M_list, z, logits, alphas



# -------------------------------
# Train / Evaluate
# -------------------------------
def train_epoch(
    model: MultiModalCollectiveMFDeep,
    loader: DataLoader,
    mu: float,
    criterion: nn.Module,
    opt: optim.Optimizer,
):
    model.train()
    sums = {'total':0., 'recon':0., 'cls':0.}
    correct = 0
    total = 0

    device = next(model.parameters()).device

    for idxs, A_batch_list, y_batch in loader:
        idxs = idxs.to(device)
        A_batch_list = [A.to(device) for A in A_batch_list]
        y_batch = y_batch.to(device)

        opt.zero_grad()
        recon_list, S_b, M_list, z, y_pred, alphas = model(idxs, A_batch_list)

        # Losses
        loss_recon = 0.0
        for v in range(len(A_batch_list)):
            loss_recon = loss_recon + mu * torch.mean((A_batch_list[v] - recon_list[v]) ** 2)
        loss_cls = criterion(y_pred, y_batch)
        loss = (loss_recon + loss_cls)

        loss.backward()
        opt.step()

        bs = y_batch.size(0)
        sums['total'] += loss.item() * bs
        sums['recon'] += float(loss_recon.detach()) * bs
        sums['cls']   += float(loss_cls.detach())   * bs

        preds = y_pred.argmax(dim=1)
        labels = y_batch
        correct += (preds == labels).sum().item()
        total += bs

    return sums['total']/max(total,1), sums, correct/max(total,1)


@torch.no_grad()
def evaluate(
    model: MultiModalCollectiveMFDeep,
    loader: DataLoader,
    mu: float,
    criterion: nn.Module,
):
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_probs = []
    all_labels = []

    device = next(model.parameters()).device

    for idxs, A_batch_list, y_batch in loader:
        idxs = idxs.to(device)
        A_batch_list = [A.to(device) for A in A_batch_list]
        y_batch = y_batch.to(device)

        recon_list, S_b, M_list, z, y_pred, alphas = model(idxs, A_batch_list)

        loss_recon = 0.0
        for v in range(len(A_batch_list)):
            loss_recon = loss_recon + mu * torch.mean((A_batch_list[v] - recon_list[v]) ** 2)
        loss_cls = criterion(y_pred, y_batch)
        loss = loss_recon + loss_cls

        bs = y_batch.size(0)
        test_loss += loss.item() * bs

        probs = torch.softmax(y_pred, dim=-1)
        preds = probs.argmax(dim=1)
        labels = y_batch
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

        test_correct += (preds == labels).sum().item()
        test_total   += bs

    avg_loss = test_loss / max(test_total,1)
    acc = test_correct / max(test_total,1)

    auc = None
    # Only compute AUC for binary classification (num_classes==2)
    if model.num_classes == 2:
        try:
            y_true = torch.cat(all_labels).numpy()
            y_score = torch.cat(all_probs).numpy()[:,1]  # prob of class 1
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = None
    return avg_loss, acc, auc


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--mu', type=float, default=1, help='reconstruction weight')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--widths', type=str, default='30, 20, 10', help='comma-separated hidden widths (same for all views)')
    parser.add_argument('--dataset', type=str, default='BP', help='Name for util.loadDataset; ignored if util missing')

    args = parser.parse_args()

    print("Dataset:", args.dataset, "\t mu= ", args.mu, "\t lr= ", args.lr, f"\t latents= {args.widths}",)

    # Load data
    A_list, y = loadDataset(args.dataset)  # either real util or synthetic fallback

    # Types and classes
    num_classes = len(torch.unique(y))
    A_list = [A.type(torch.float32).to(args.device) for A in A_list]
    y = y.type(torch.long)

    # Split
    n_samples = A_list[0].shape[0]
    train_idx, test_idx = train_test_split(torch.arange(n_samples), test_size=0.2)

    dataset = MultiViewMatrixDataset(A_list, y)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=100, shuffle=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=100, shuffle=True)

    # Parse widths
    widths = None
    if args.widths.strip():
        widths = [int(x) for x in args.widths.split(',') if x.strip()]

    # Build model
    model = MultiModalCollectiveMFDeep(
        [A for A in A_list],
        n_samples=n_samples,
        num_classes=num_classes,
        widths=widths,
        layers=len(widths),
    ).to(args.device)

    model._init_W_cache = [W.to(args.device) for W in model._init_W_cache]

    # Criterion / Optimizer
    criterion = nn.CrossEntropyLoss()
    opt = optim.ASGD(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(1, args.epochs + 1):

        tr_loss, parts, tr_acc = train_epoch(model, train_loader, args.mu, criterion, opt)

        if epoch % 100 == 0 or epoch == 1 or epoch == args.epochs:
            te_loss, te_acc, te_auc = evaluate(model, test_loader, args.mu, criterion)
            auc_str = f"{te_auc:.4f}" if te_auc is not None else "NA"
            print(
                f"Epoch {epoch:4d} | train loss {tr_loss:.4f} "
                f"(recon {parts['recon']/max(len(train_loader.dataset),1):.6f}, "
                f"cls {parts['cls']/max(len(train_loader.dataset),1):.6f}) "
                f"| train acc {tr_acc:.3f} | test loss {te_loss:.4f} | test acc {te_acc:.3f} | AUC {auc_str}"
            )

    # Final evaluation
    te_loss, te_acc, te_auc = evaluate(model, test_loader, args.mu, criterion)
    auc_str = f"{te_auc:.4f}" if te_auc is not None else "NA"
    print(f"FINAL TEST — ACC: {te_acc:.4f} | AUC: {auc_str}")


if __name__ == "__main__":
    main()
