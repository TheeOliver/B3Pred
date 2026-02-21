import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

import torch
from configs.predictor_config import TrainConfig as train_settings
from scripts.evaluate import test_model

# ---------------------------------------------------------------------------
# Global device: use GPU if available, otherwise fall back to CPU
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Using device: {DEVICE}")


def train_model(model, train_loader, val_loader, epochs, target_labels, loss_type: str = 'crossentropy',
                learning_rate: float = 0.001, hetero=False, log=False, save_to=None):

    if log and not WANDB_AVAILABLE:
        print("Warning: wandb logging requested but wandb not installed. Disabling logging.")
        log = False

    print('Starting training')
    criterion = train_settings.loss_function[loss_type]()

    # Move model to GPU
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    best_f1 = -1.0

    for epoch in range(epochs):
        losses = []
        model.train()
        i = 1
        for data in train_loader:
            data = data.to(DEVICE)

            optimizer.zero_grad()
            out = model(data)  # predict per graph

            i += 1
            loss = criterion(out, data.y.view(-1))  # compare graph prediction vs label
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        res = test_model(val_loader, model, target_labels, hetero=hetero)
        res['epoch'] = epoch
        epoch_loss = np.mean(losses) if losses else 0
        # epoch_loss = sum(losses) / len(losses) if len(losses) > 0 else 0
        print(epoch_loss)
        res['epoch_loss'] = epoch_loss
        print(res)

        if log and WANDB_AVAILABLE:
            wandb.log(res)

        if save_to is not None:
            if best_f1 is None or res['macro_f1'] > best_f1:
                torch.save(model.state_dict(), save_to)
                best_f1 = res['macro_f1']

    return res, model