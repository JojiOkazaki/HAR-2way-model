import torch

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path

        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_model(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def save_model(self, model):
        torch.save(model.state_dict(), self.path)
