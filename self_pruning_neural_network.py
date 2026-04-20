
import os, time, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision, torchvision.transforms as transforms

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 3.0))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gate_values(self):
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

class SelfPruningNet(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = PrunableLinear(256, 10)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

def compute_sparsity_loss(model):
    total = torch.tensor(0.0, device=DEVICE)
    for layer in model.prunable_layers():
        total = total + torch.sigmoid(layer.gate_scores).sum()
    return total

def get_cifar10_loaders(batch_size=128):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR10("./data", True, download=True, transform=tr)
    test_ds  = torchvision.datasets.CIFAR10("./data", False, download=True, transform=te)
    return (torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True,
                                        num_workers=2, pin_memory=True),
            torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        num_workers=2, pin_memory=True))

def train_one_epoch(model, loader, optimizer, criterion, lam):
    model.train(); running = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        loss = criterion(model(imgs), labels) + lam * compute_sparsity_loss(model)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running += loss.item()
    return running / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def compute_sparsity_pct(model, thr=1e-2):
    total = pruned = 0
    for layer in model.prunable_layers():
        g = layer.get_gate_values()
        total += g.numel(); pruned += (g < thr).sum().item()
    return 100.0 * pruned / total if total else 0.0

@torch.no_grad()
def collect_gates(model):
    return np.concatenate([layer.get_gate_values().cpu().numpy().ravel()
                           for layer in model.prunable_layers()])

def plot_gate_dist(gates, lam, path):
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=100, color="#2196F3", edgecolor="white", alpha=0.85)
    plt.title(f"Gate Value Distribution (lambda={lam})", fontsize=15, fontweight="bold")
    plt.xlabel("sigmoid(gate_score)"); plt.ylabel("Count")
    plt.axvline(0.01, color="red", ls="--", lw=1.5, label="Prune threshold (0.01)")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved: {path}")

def plot_comparison(results, path):
    lams = [r["lambda"] for r in results]
    accs = [r["accuracy"] for r in results]
    spars = [r["sparsity"] for r in results]
    x = np.arange(len(lams)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 6))
    b1 = ax1.bar(x - w/2, accs, w, label="Accuracy (%)", color="#4CAF50")
    ax1.set_ylabel("Accuracy (%)", color="#4CAF50")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w/2, spars, w, label="Sparsity (%)", color="#FF5722")
    ax2.set_ylabel("Sparsity (%)", color="#FF5722")
    ax1.set_xlabel("Lambda"); ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l:.0e}" for l in lams])
    ax1.set_title("Accuracy vs Sparsity", fontsize=15, fontweight="bold")
    for b in b1:
        ax1.annotate(f"{b.get_height():.1f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                     textcoords="offset points", xytext=(0,4), ha="center", fontsize=9)
    for b in b2:
        ax2.annotate(f"{b.get_height():.1f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                     textcoords="offset points", xytext=(0,4), ha="center", fontsize=9)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper center")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved: {path}")

def run_experiment(lam, train_loader, test_loader, epochs=25, lr=1e-3):
    print(f"\n{'='*60}\n  EXPERIMENT | lambda = {lam}\n{'='*60}")
    model = SelfPruningNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(1, epochs+1):
        loss = train_one_epoch(model, train_loader, opt, crit, lam)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            acc = evaluate(model, test_loader)
            sp = compute_sparsity_pct(model)
            print(f"  Ep {ep:3d}/{epochs} | Loss {loss:.4f} | "
                  f"Acc {acc:.2f}% | Sparsity {sp:.2f}% | {time.time()-t0:.0f}s")
    acc = evaluate(model, test_loader)
    sp = compute_sparsity_pct(model)
    gv = collect_gates(model)
    print(f"  >> Final Acc: {acc:.2f}% | Sparsity: {sp:.2f}%")
    print(f"  >> Gates near 0: {(gv<0.01).sum():,} | near 1: {(gv>0.99).sum():,}")
    return {"lambda": lam, "accuracy": acc, "sparsity": sp, "model": model, "gate_values": gv}

def main():
    print("="*60)
    print("  SELF-PRUNING NEURAL NETWORK - CIFAR-10")
    print("  Tredence Analytics Case Study")
    print("="*60)

    LAMBDAS = [1e-5, 5e-4, 1e-2]
    OUT = "./results"; os.makedirs(OUT, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders()
    results, best = [], None

    for lam in LAMBDAS:
        r = run_experiment(lam, train_loader, test_loader)
        results.append(r)
        if best is None or r["accuracy"] > best["accuracy"]:
            best = r

    print(f"\n{'='*60}\n  RESULTS SUMMARY\n{'='*60}")
    print(f"  {'Lambda':<14}{'Accuracy (%)':<20}{'Sparsity (%)':<20}")
    print(f"  {'-'*14}{'-'*20}{'-'*20}")
    for r in results:
        print(f"  {r['lambda']:<14.1e}{r['accuracy']:<20.2f}{r['sparsity']:<20.2f}")

    plot_gate_dist(best["gate_values"], best["lambda"],
                   os.path.join(OUT, "gate_distribution_best.png"))
    for r in results:
        tag = f"{r['lambda']:.0e}".replace("-","neg")
        plot_gate_dist(r["gate_values"], r["lambda"],
                       os.path.join(OUT, f"gate_dist_{tag}.png"))
    plot_comparison(results, os.path.join(OUT, "accuracy_vs_sparsity.png"))

    with open(os.path.join(OUT, "results_summary.txt"), "w") as f:
        f.write("Self-Pruning Neural Network - Results\n" + "="*50 + "\n")
        for r in results:
            f.write(f"lambda={r['lambda']:.1e}  acc={r['accuracy']:.2f}%  "
                    f"sparsity={r['sparsity']:.2f}%\n")

    print(f"\n[INFO] All outputs saved to '{OUT}/' - Done!")

if __name__ == "__main__":
    main()
