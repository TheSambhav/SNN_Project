import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import TradingSNN

def train_model(X_train, Y_train, num_epochs=20, batch_size=32, time_steps=20):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

    model = TradingSNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch, time_steps=time_steps)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        acc = correct / total
        epoch_loss = total_loss / total
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {acc:.2%}")

    return model