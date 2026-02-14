import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path('DAPWH')
SPLIT_DIR = BASE_DIR / 'DAPWH_Final_Split'

# to use GPU
DEVICE = torch.device("cuda:0") 
torch.cuda.set_device(DEVICE)
EPOCHS = 150
BATCH_SIZE = 64
IMG_SIZE = 224

# list final results
final_results = []

def evaluate_torch_model(model, test_loader, model_name, train_time):
    """Compute metrics on the test set for PyTorch models."""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    return {
        'model': model_name,
        'acccuracy': acc,
        'precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'time_train': train_time
    }

def train_torch_model(model_name, weights_config, num_classes, dataloaders, dataset_sizes):
    start_time = time.time()
    print(f"\n starting traning: {model_name}")
    
    model = getattr(models, model_name)(weights=weights_config)
    
    # Ajuste das Heads
    if 'resnet' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vit' in model_name:
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif 'convnext' in model_name:
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif 'efficientnet' in model_name:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    history = []
    plt.ion()

    for epoch in range(EPOCHS):
        epoch_log = {'epoch': epoch + 1}
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=f"Ep {epoch+1} {phase}", leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_log[f'{phase}_loss'] = running_loss / dataset_sizes[phase]
            epoch_log[f'{phase}_acc'] = running_corrects.double().item() / dataset_sizes[phase]
        
        history.append(epoch_log)
        clear_output(wait=True)
        df_hist = pd.DataFrame(history)
        print(f"Modelo: {model_name} | Época: {epoch+1}/{EPOCHS}")
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].plot(df_hist['train_loss'], label='Train'); ax[0].plot(df_hist['val_loss'], label='Val'); ax[0].legend(); ax[0].set_title("Loss")
        ax[1].plot(df_hist['train_acc'], label='Train'); ax[1].plot(df_hist['val_acc'], label='Val'); ax[1].legend(); ax[1].set_title("Accuracy")
        plt.show(); plt.pause(0.1)

    plt.ioff()
    train_time = time.time() - start_time
    torch.save(model.state_dict(), f"{model_name}_final.pth")
    
    # final valuation on test
    metrics = evaluate_torch_model(model, dataloaders['test'], model_name, train_time)
    final_results.append(metrics)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

phases = ['train', 'val', 'test']
image_datasets = {x: datasets.ImageFolder(SPLIT_DIR / x, transform) for x in phases}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train')) for x in phases}
dataset_sizes = {x: len(image_datasets[x]) for x in phases}
num_classes = len(image_datasets['train'].classes)

# model loop
MODEL_LIST = [
    ('efficientnet_v2_s', 'torch', 'DEFAULT'),
    ('convnext_tiny',     'torch', 'DEFAULT'),
    ('vit_b_16',          'torch', 'DEFAULT'),
    ('vgg16',             'torch', 'DEFAULT'),
    ('resnet50',          'torch', 'DEFAULT'),
    ('yolov12s-cls.pt',    'yolo',  None)
]

def get_gpu_manager():
    """Retorna o status e limpa a memória da GPU."""
    if torch.cuda.is_available():
        gc.collect()
        # clear any cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        free, total = torch.cuda.mem_get_info()
        print(f"\n[GPU] Memory Clear | Free: {free/1024**3:.2f}GB / {total/1024**3:.2f}GB")
    else:
        print("\n[WARNING] GPU not detected. Using CPU.")

def monitor_gpu_during_train():
    """VRAM usage"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_reserved() / 1024**3
        return f"VRAM: {used:.2f}GB (Pico: {peak:.2f}GB)"
    return "CPU Mode"

final_comparison = []

def train_torch_model(model_name, weights_config, num_classes, dataloaders, dataset_sizes):
    start_time = time.time()
    print(f"\n>>> Starting: {model_name}")
    
    model = getattr(models, model_name)(weights=weights_config)
    if 'resnet' in model_name: 
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vit' in model_name: 
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif 'convnext' in model_name: 
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif 'efficientnet' in model_name: 
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'vgg' in model_name:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epoch_history = []

    for epoch in range(EPOCHS):
        stats = {'epoch': epoch + 1, 'model': model_name}
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{model_name} Ep {epoch+1}", leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward(); optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            stats[f'{phase}_loss'] = running_loss / dataset_sizes[phase]
            stats[f'{phase}_acc'] = (running_corrects.double() / dataset_sizes[phase]).item()
        
        epoch_history.append(stats)
        
        # Feedback
        clear_output(wait=True)
        display(pd.DataFrame(epoch_history).tail(1))
    save_dir = Path("trained_models")
    save_dir.mkdir(exist_ok=True)
    
    model_path = save_dir / f"{model_name}_final.pth"
    torch.save(model.state_dict(), model_path)
    # final csv per epoch
    df_history = pd.DataFrame(epoch_history)
    csv_path = f"traning_results/{model_name}_training_history.csv"
    df_history.to_csv(csv_path, index=False)
    print(f"✓ Histórico salvo: {csv_path}")

    # final valuation on test
    train_duration = time.time() - start_time
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy()); y_pred.extend(preds.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    final_comparison.append({
        'model': model_name, 'test_accuracy': acc, 'test_f1': f1, 'time_train': train_duration, 'test_prec':prec, 'test_rec':rec
    })

# --- traning loop
for name, framework, config in MODEL_LIST:
    get_gpu_manager() 
    
    if framework == 'torch':
        train_torch_model(name, config, num_classes, dataloaders, dataset_sizes)
    
    elif framework == 'yolo':
        start_yolo = time.time()
        yolo_model = YOLO(name)
        results = yolo_model.train(
            data=str(SPLIT_DIR), epochs=EPOCHS, imgsz=IMG_SIZE, 
            device=0
        )   
        final_comparison.append({
            'model': name, 
            'test_f1': results.results_dict.get('metrics/accuracy_top1', 0),
            'time_train': (time.time()-start_yolo)
        })

# save the final table
pd.DataFrame(final_comparison).to_csv("traning_results/final_results_table_test.csv", index=False)
