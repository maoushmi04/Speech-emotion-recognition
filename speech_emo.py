
"""
Speech emotion recognition on TESS using a pretrained VGG16 (transfer learning).
 
Usage:
    python speech_emo.py                      # train + evaluate
    python speech_emo.py path\\to\\audio.wav    # predict with a saved model
 
Edit DATA_PATH below to where your TESS folder is.
"""
import os
import re
import sys
import random
from urllib import response
 
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt
 
# ----------------------------- CONFIG ---------------------------------------
DATA_PATH = r"C:\data\TESS"
MODEL_PATH = "emotion_recognition_model.pth"
EMOTIONS = ["anger", "disgust", "fear", "happiness", "pleasant_surprise", "sadness", "neutral"]
SR = 16000
N_MELS = 128
MAX_LEN = 128          # time frames
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
SEED = 42
UNFREEZE_LAST_BLOCK = True   # fine-tune the last VGG conv block too
# -----------------------------------------------------------------------------
 
# Folder-name suffix (lowercase) -> emotion. Handles the different TESS spellings.
ALIASES = {
    "angry": "anger", "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness", "happiness": "happiness",
    "pleasant_surprise": "pleasant_surprise", "pleasant_surprised": "pleasant_surprise",
    "ps": "pleasant_surprise",
    "sad": "sadness", "sadness": "sadness",
    "neutral": "neutral",
}
 
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
 
 
def audio_to_tensor(path):
    """Audio file -> normalized 3x128x128 tensor. Used for training AND prediction."""
    y, sr = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    db = librosa.power_to_db(mel, ref=np.max)          # range roughly [-80, 0]
    db = np.clip((db + 80.0) / 80.0, 0.0, 1.0)          # -> [0, 1]
    if db.shape[1] < MAX_LEN:
        db = np.pad(db, ((0, 0), (0, MAX_LEN - db.shape[1])), mode="constant")
    else:
        db = db[:, :MAX_LEN]
    x = torch.tensor(db, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD
 
 
def find_files(data_path):
    """Walk data_path, find OAF_*/YAF_* folders, return (file_path, label_idx, word)."""
    items = []
    counts = {e: 0 for e in EMOTIONS}
    for root, dirs, files in os.walk(data_path):
        m = re.match(r"^(OAF|YAF)_(.+)$", os.path.basename(root), re.IGNORECASE)
        if not m:
            continue
        emotion = ALIASES.get(m.group(2).lower())
        if emotion is None:
            print(f"  [skip] unknown folder: {root}")
            continue
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            parts = os.path.splitext(f)[0].split("_")
            word = parts[1].lower() if len(parts) >= 3 else parts[0].lower()
            items.append((os.path.join(root, f), EMOTIONS.index(emotion), word))
            counts[emotion] += 1
    print("Files per emotion:", counts)
    return items
 
 
class EmotionDataset(Dataset):
    """Precomputes all spectrograms once and keeps them in memory (much faster)."""
 
    def __init__(self, items):
        self.x, self.y = [], []
        for i, (path, label, _) in enumerate(items):
            self.x.append(audio_to_tensor(path))
            self.y.append(label)
            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{len(items)} files")
        self.x = torch.stack(self.x)
        self.y = torch.tensor(self.y, dtype=torch.long)
 
    def __len__(self):
        return len(self.y)
 
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
 
 
def split_by_word(items, seed=SEED, train_frac=0.7, val_frac=0.15):
    """Split so the same spoken word never appears in two splits (avoids leakage)."""
    words = sorted({w for _, _, w in items})
    random.Random(seed).shuffle(words)
    n_train = int(train_frac * len(words))
    n_val = int(val_frac * len(words))
    split_of = {}
    for i, w in enumerate(words):
        split_of[w] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    out = {"train": [], "val": [], "test": []}
    for it in items:
        out[split_of[it[2]]].append(it)
    return out["train"], out["val"], out["test"]
 
 
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for p in self.vgg.parameters():
            p.requires_grad = False
        if UNFREEZE_LAST_BLOCK:
            for p in self.vgg.features[24:].parameters():
                p.requires_grad = True
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)
 
    def forward(self, x):
        return self.vgg(x)
 
 
def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)
 
 
def predict_emotion(audio_path, model=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = EmotionRecognitionModel(len(EMOTIONS)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    x = audio_to_tensor(audio_path).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return EMOTIONS[idx], float(probs[idx])
 
 
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
 
    print("Scanning:", DATA_PATH)
    items = find_files(DATA_PATH)
    assert len(items) > 0, "No .wav files found - check DATA_PATH"
 
    train_items, val_items, test_items = split_by_word(items)
    print(f"Split (by word): train={len(train_items)} val={len(val_items)} test={len(test_items)}")
 
    print("Computing spectrograms (one time)...")
    train_ds = EmotionDataset(train_items)
    val_ds = EmotionDataset(val_items)
    test_ds = EmotionDataset(test_items)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
 
    model = EmotionRecognitionModel(len(EMOTIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
 
    history = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}
    best_val = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        history["tr_loss"].append(tr_loss); history["tr_acc"].append(tr_acc)
        history["va_loss"].append(va_loss); history["va_acc"].append(va_acc)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc >= best_val:
            best_val = va_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  saved best model")
 
    # Test with the best checkpoint
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    te_loss, te_acc = run_epoch(model, test_loader, criterion, device)
    print(f"\nTest loss {te_loss:.4f}, Test accuracy {te_acc:.4f}")
 
    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["tr_loss"], label="train"); ax[0].plot(history["va_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(history["tr_acc"], label="train"); ax[1].plot(history["va_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Saved training_curves.png")
 
    # Demo prediction on one test file (unseen words)
    if test_items:
        path, label, _ = test_items[0]
        pred, conf = predict_emotion(path, model, device)
        print(f"\nDemo: {os.path.basename(path)} | true={EMOTIONS[label]} | "
              f"predicted={pred} ({conf:.2%})")
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        emo, conf = predict_emotion(sys.argv[1])
        print(f"Predicted Emotion: {emo} ({conf:.2%})")
    else:
        main()
 

