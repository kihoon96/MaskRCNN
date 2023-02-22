import os
import argparse
import torch
import numpy as np
import wandb

setwandb = False
if setwandb:
    wandb.init(project="MRCNN")
    wandb.config = {
        "epochs" : cfg.epochs,
        "batch_size" : cfg.batch_size
    }

