import sys
import numpy as np
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import lovely_tensors as lt
lt.monkey_patch()
from peft import LoraConfig, get_peft_model
import timm
import lpips
from torch.nn import functional as F
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from torch.utils.data import Dataset, DataLoader
import h5py
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from torchmetrics import Accuracy
import optuna
import pickle
from tqdm import tqdm

num_classes=30
device = 'cpu'
batch_size = 2048
num_workers=8
ckpt_path = None
dataset_path = '/Users/rotemisraeli/Documents/datasets/plants/'
interpolation = transforms.InterpolationMode.NEAREST
train_features_path = '/Users/rotemisraeli/Documents/datasets/plants/train_features-L.pth'
test_features_path = '/Users/rotemisraeli/Documents/datasets/plants/test_features-L.pth'

def set_seed(seed=42):
    np.random.seed(seed)  # 🎲 Set seed for NumPy
    torch.manual_seed(seed)  # 🚀 Set seed for PyTorch on CPU
    torch.cuda.manual_seed(seed)  # 🚀 Set seed for PyTorch on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class SimpleMLP(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], output_size=512, dropout_rate=0.2):
        super(SimpleMLP, self).__init__()
        self.layers = nn.ModuleList()

        # Dynamically create hidden layers
        last_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(last_size, size))
            self.layers.append(nn.BatchNorm1d(size))  # Batch normalization
            self.layers.append(nn.SiLU())
            self.layers.append(nn.Dropout(dropout_rate))  # Dropout
            last_size = size

        # Output layer
        self.output = nn.Linear(last_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x

class FeatureDataset(Dataset):
    def __init__(self, feature_file):
        self.features, self.labels = torch.load(feature_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_model(model_name,layer_sizes,latent_size):
    if model_name=='mlp':
        mlp = SimpleMLP(latent_size,layer_sizes,num_classes)
        kan = torch.nn.Identity()
        return mlp,kan

    mlp = SimpleMLP(latent_size,[],layer_sizes[0])

    if model_name=='kan':
        from kan import KAN
        kan = KAN(width=[layer_sizes[0],layer_sizes[1],num_classes], grid=5, k=3, seed=0)
    elif model_name=='efficient-kan':
        sys.path.insert(0,'efficient-kan/src/')
        import efficient_kan
        kan = efficient_kan.kan.KAN(layers_hidden=[layer_sizes[0],layer_sizes[1],num_classes])
    elif model_name=='FastKAN':
        # git clone https://github.com/ZiyaoLi/fast-kan.git
        sys.path.insert(0,'fast-kan/fastkan')
        from fastkan import FastKAN
        kan = FastKAN(layers_hidden=[layer_sizes[0],layer_sizes[1],num_classes])
    elif model_name=='ChebyKAN':
        # git clone https://github.com/SynodicMonth/ChebyKAN.git
        from ChebyKAN.ChebyKANLayer import ChebyKANLayer
        kan = nn.Sequential(
                nn.LayerNorm(layer_sizes[0]),
                ChebyKANLayer(layer_sizes[0], layer_sizes[1],4),
                nn.LayerNorm(layer_sizes[1]),
                ChebyKANLayer(layer_sizes[1], num_classes,4)
            )
    elif model_name=='JacobiKAN':
        # git clone https://github.com/SpaceLearner/JacobiKAN.git
        from JacobiKAN.JacobiKANLayer import JacobiKANLayer
        kan = nn.Sequential(
                nn.LayerNorm(layer_sizes[0]),
                JacobiKANLayer(layer_sizes[0], layer_sizes[1],4),
                nn.LayerNorm(layer_sizes[1]),
                JacobiKANLayer(layer_sizes[1], num_classes,4)
            )
    elif model_name=='RBFKAN':
        # git clone https://github.com/sidhu2690/RBF-KAN.git
        sys.path.insert(0,'RBF-KAN')
        from RBF_KAN import RBFKAN
        kan = RBFKAN([layer_sizes[0],layer_sizes[1],num_classes])
    elif model_name=='fcn_kan':
        # git clone https://github.com/Zhangyanbo/FCN-KAN.git
        sys.path.insert(0,'FCN-KAN/')
        from kan_layer import KANLayer
        kan = nn.Sequential(
                KANLayer(layer_sizes[0], layer_sizes[1]),
                KANLayer(layer_sizes[1], num_classes)
            )
    elif model_name=='FourierKAN':
        # git clone https://github.com/GistNoesis/FourierKAN.git
        from FourierKAN.fftKAN import NaiveFourierKANLayer
        kan = nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(layer_sizes[0]),
                NaiveFourierKANLayer(layer_sizes[0], layer_sizes[1],5),
                nn.SiLU(),
                nn.LayerNorm(layer_sizes[1]),
                NaiveFourierKANLayer(layer_sizes[1], num_classes,5)
            )

    return mlp,kan

class Lightning_Model(pl.LightningModule):
    def __init__(self, model_name,layer_sizes,latent_size,lr):
        super(Lightning_Model, self).__init__()
        self.lr = lr
        self.mlp, self.kan = get_model(model_name,layer_sizes,latent_size)
        self.checkpoint_interval = 50000
        self.cross_entropy = nn.CrossEntropyLoss()
        self.train_acc = Accuracy('MULTICLASS',num_classes=30)
        self.val_acc = Accuracy('MULTICLASS',num_classes=30)


    def load_ckpt(self,ckpt_path):
        old_dict = torch.load(ckpt_path)
        new_dict = {}
        for key in old_dict.keys():
            new_dict['.'.join(key.split('.')[1:])] = old_dict[key]
        self.model.load_state_dict(new_dict)

    def forward(self, x,batch_idx):
        x = self.mlp(x)
        x = self.kan(x)
        return x

    def training_step(self, batch, batch_idx):
        features, labels = batch
        out = self.forward(features,(batch_idx,self.trainer.current_epoch))

        cross_entropy = self.cross_entropy(out,labels)
        self.log("cross_entropy", cross_entropy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc(out,labels), prog_bar=True)


        if (batch_idx + 1) % self.checkpoint_interval == 0:
            self.save_custom_checkpoint(batch_idx)
        return cross_entropy

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        out = self.forward(features,(batch_idx,self.trainer.current_epoch))

        cross_entropy = self.cross_entropy(out,labels)
        self.log("val_cross_entropy", cross_entropy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc(out,labels), prog_bar=True)

        if (batch_idx + 1) % self.checkpoint_interval == 0:
            self.save_custom_checkpoint(batch_idx)
        return cross_entropy

    def save_custom_checkpoint(self, batch_idx):
        checkpoint_filename = f'last_pp.ckpt'
        checkpoint_path = os.path.join(self.trainer.default_root_dir, 'checkpoints', checkpoint_filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.trainer.save_checkpoint(checkpoint_path)

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
        return optimizer

    def train_dataloader(self):
        train_dataset = FeatureDataset(train_features_path)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=num_workers, pin_memory=False, persistent_workers=True)

    def val_dataloader(self):
        val_dataset = FeatureDataset(test_features_path)
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=False, persistent_workers=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_latent_size():
    return torch.load(test_features_path)[0].shape[-1]





import optuna
from itertools import product
import csv
import time
from itertools import product
from tqdm import tqdm
import optuna
import pytorch_lightning as pl

def objective(params):
    model_name, lr, size1, size2 = params
    layer_sizes = [size1, size2]
    latent_size = extract_latent_size()

    model = Lightning_Model(model_name, layer_sizes, latent_size, lr)
    num_trainable_params = count_parameters(model)

    logger = pl.loggers.CSVLogger(save_dir='logs/', name=model_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_cross_entropy',
        dirpath='checkpoints/',
        filename=f'{model_name}-{{epoch:02d}}-{{val_cross_entropy:.4f}}-{{val_acc:.4f}}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=33,
        precision='16-mixed'
    )
    start_time = time.time()
    trainer.fit(model)
    training_time = time.time() - start_time

    val_loss = trainer.callback_metrics.get("val_cross_entropy", float('inf'))
    return num_trainable_params, training_time, val_loss

def main():
    param_grid = {
        'model_name': ['mlp', 'efficient-kan', 'FastKAN', 'ChebyKAN', 'JacobiKAN', 'RBFKAN'],
        'lr': [0.01],
        'n_units_l1': [24],
        'n_units_l2': [16]
    }

    all_params = list(product(param_grid['model_name'], param_grid['lr'], param_grid['n_units_l1'], param_grid['n_units_l2']))
    total_trials = len(all_params)
    results = []

    for index, params in tqdm(enumerate(all_params, start=1)):
        num_trainable_params, training_time, result = objective(params)
        results.append((params, num_trainable_params, training_time, result))
        print("\n========================================")
        print(f"Trial {index}/{total_trials} Completed")
        print(f"Params: {params}, Trainable Params: {num_trainable_params}, Time to Train: {training_time:.2f} sec, Result: {result}")
        print("========================================\n")

    # Saving results to a CSV file
    with open('training_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['model_name', 'n_units_l1', 'n_units_l2', 'trainable_params', 'time_to_train', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for params, num_params, train_time, val_loss in results:
            writer.writerow({
                'model_name': params[0],
                'n_units_l1': params[2],
                'n_units_l2': params[3],
                'trainable_params': num_params,
                'time_to_train': train_time,
                'val_loss': val_loss
            })

    # Find and print the best trial based on validation loss
    best_trial = min(results, key=lambda x: x[3])
    print("Best trial:")
    print(f"Params: {best_trial[0]}, Loss: {best_trial[3]}")

if __name__ == '__main__':
    main()
