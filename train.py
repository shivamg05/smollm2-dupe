import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import MLP

def make_data(n=512, in_dim=8, out_dim=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    w = torch.randn(in_dim, out_dim, generator=g)
    y = x @ w + 0.1*torch.randn(n, out_dim, generator=g)
    return x, y

def get_loaders(x, y, batch_size=64, split=0.8):
    n = x.size(0)
    split_idx = math.floor(n * split)

    x_train, y_train = x[:split_idx], y[:split_idx]
    x_val, y_val = x[split_idx:], y[split_idx:]

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0)
    
    return total / len(loader.dataset)

def main():
    in_dim = 8
    emb_dim = 32
    out_dim = 3
    lr = 1e-3
    epochs = 1000
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_data(n=512, in_dim=in_dim, out_dim=out_dim, seed=0)
    train_loader, val_loader = get_loaders(x, y, batch_size=batch_size)

    model = MLP(in_dim=in_dim, emb_dim=emb_dim, out_dim=out_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        print(f"epoch: {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

if __name__ == "__main__":
    main()

