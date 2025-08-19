import os
import math
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
BLANK = 0  
char2idx = {c: i + 1 for i, c in enumerate(ALPHABET)}
idx2char = {i + 1: c for i, c in enumerate(ALPHABET)}
idx2char[BLANK] = ""  

def encode_label(text: str) -> List[int]:
  
    text = text.strip().upper()
    return [char2idx[c] for c in text if c in char2idx]

def ctc_greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor) -> List[str]:
   
    with torch.no_grad():
        probs = logits.log_softmax(2).argmax(2)  # (T, B)
        probs = probs.cpu().numpy().T  # (B, T)

    results = []
    for b, L in enumerate(input_lengths.cpu().numpy()):
        prev = BLANK
        s = []
        for t in range(L):
            p = int(probs[b, t])
            if p != prev and p != BLANK:
                s.append(idx2char.get(p, ""))
            prev = p
        results.append("".join(s))
    return results

class PlateDataset(Dataset):
    def __init__(self, csv_path: str, img_h: int, img_w: int):
        self.df = pd.read_csv(csv_path)
        if not {"image_path", "label"}.issubset(self.df.columns):
            raise ValueError("CSV must have columns: image_path,label")
        self.img_h = img_h
        self.img_w = img_w

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = str(row["label"]).strip()

        img = Image.open(path).convert("RGB")
        img = self.transform(img)  # (1, H, W)

        target = torch.tensor(encode_label(label), dtype=torch.long)
        return img, target, label, path


def crnn_collate(batch, downsample_factor: int = 4):
  
    imgs, targets, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # (B, 1, H, W)


    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    if target_lengths.max().item() == 0:
        raise ValueError("Found empty label; remove or fix entries with blank labels.")
    targets_concat = torch.cat(targets, dim=0)

    B, C, H, W = imgs.size()
    T = W // downsample_factor
    input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)

    return imgs, targets_concat, input_lengths, target_lengths, labels, paths

class CRNN(nn.Module):
 
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 2), nn.ReLU(True),  # make H close to 1
        )
        self.rnn = nn.LSTM(
            input_size=256, hidden_size=256, num_layers=2,
            bidirectional=True, batch_first=False, dropout=0.1
        )
        self.fc = nn.Linear(512, num_classes)  # bi-directional => 2*hidden

    def forward(self, x):  # x: (B, 1, H, W)
        feats = self.cnn(x)  # (B, 256, H', W')
        B, C, Hp, Wp = feats.size()
        assert Hp == 1 or Hp == 2, 
        feats = feats.squeeze(2)               
        feats = feats.permute(2, 0, 1).contiguous() 
        seq, _ = self.rnn(feats)              
        logits = self.fc(seq)                  
        return logits

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(loader, desc="Train", ncols=100)
    for imgs, targets, input_lengths, target_lengths, _, _ in pbar:
        imgs = imgs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        logits = model(imgs)  # (T, B, C)
        log_probs = logits.log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_pred, all_gt = [], []
    for imgs, targets, input_lengths, target_lengths, labels, _ in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(imgs)
        log_probs = logits.log_softmax(2)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        epoch_loss += loss.item()

        # simple accuracy on full-string equality (greedy decode)
        preds = ctc_greedy_decode(logits, input_lengths)
        all_pred.extend(preds)
        all_gt.extend(labels)

    exact = np.mean([p == g for p, g in zip(all_pred, all_gt)]) if all_gt else 0.0
    return epoch_loss / max(1, len(loader)), exact, list(zip(all_gt[:10], all_pred[:10]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--img_h", type=int, default=32)
    parser.add_argument("--img_w", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PlateDataset(args.train_csv, args.img_h, args.img_w)
    val_ds   = PlateDataset(args.val_csv,   args.img_h, args.img_w)

    collate = lambda b: crnn_collate(b, downsample_factor=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)

    num_classes = 1 + len(ALPHABET)  # +1 for CTC blank
    model = CRNN(num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, vacc, samples = validate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch}/{args.epochs} | train_loss={tl:.4f} "
              f"| val_loss={vl:.4f} | val_exact_match={vacc*100:.2f}%")
        print("Samples (GT -> Pred):")
        for gt, pr in samples:
            print(f"  {gt} -> {pr}")

        # Save best
        if vacc > best_acc:
            best_acc = vacc
            ckpt_path = os.path.join(args.out_dir, f"crnn_best_acc_{best_acc:.4f}.pth")
            torch.save({
                "model": model.state_dict(),
                "alphabet": ALPHABET,
                "img_h": args.img_h,
                "img_w": args.img_w,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.out_dir, "crnn_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final weights to {final_path}")


if __name__ == "__main__":
    main()
