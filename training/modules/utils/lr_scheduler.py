import math

def lr_lambda(epoch, warmup_epochs, total_epochs, min_lr_ratio=0.1):
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        scale = 0.5 * (1 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * scale
