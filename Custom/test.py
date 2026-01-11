import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from utils import cfg, get_device, get_model, create_dataloaders, load_model

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    # plt.show()
    print("[INFO] Matrice di confusione salvata in 'confusion_matrix.png'")

def plot_test_metrics(results):
    """
    Stampa un istogramma Accuracy per Immagine (corretto/errato).
    results: lista di booleani o int (1/0)
    """
    plt.figure(figsize=(10, 4))
    
    # È un grafico a barre dove x è l'indice dell'immagine, y è 0 o 1
    indices = range(len(results))
    plt.bar(indices, results, width=1.0)
    plt.xlabel('Image Index')
    plt.ylabel('Accuracy (1=Correct, 0=Wrong)')
    plt.title('Per-Image Accuracy on Test Set')
    plt.yticks([0, 1], ['Wrong', 'Correct'])
    
    plt.savefig('test_metrics.png')
    # plt.show()
    print("[INFO] Grafico test salvato in 'test_metrics.png'")

def test(config):
    device = get_device(config.device)
    
    # 1. Dati
    _, _, test_loader = create_dataloaders(config)
    classes = test_loader.dataset.dataset.classes # Access inner dataset classes
    
    # 2. Modello
    print(f"\n[TEST] Avvio test modello salvato...")
    
    try:
        model = get_model(config.model_name, config.input_channels, config.num_classes).to(device)
    except ValueError as e:
        print(f"[ERRORE] {e}")
        return
    
    # 3. Load Checkpoint
    # Cerca un file che matchi il pattern o usa quello in config
    # Se il file esatto non esiste, cerchiamo un best model pattern o usiamo quello di default
    # Spesso dopo il training il nome è cambiato (es. _best_acc_epoch.pth).
    # Qui usiamo quello specificato in config, l'utente deve assicurarsi che punti al file giusto 
    # oppure che train abbia sovrascritto quello base.
    
    ckpt_path = config.checkpoint_path
    
    # Check if a specific best model exists if the default one is missing
    if not os.path.exists(ckpt_path):
        # Proviamo a cercare file che iniziano con il nome
        # (Logica semplice, si potrebbe migliorare)
        dir_path = os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else "."
        base_name = os.path.basename(ckpt_path).replace(".pth", "")
        candidates = [f for f in os.listdir(dir_path) if f.startswith(base_name) and f.endswith(".pth")]
        if candidates:
            # Prendi il più recente o quello con acc più alta?
            # Semplifichiamo prendendo l'ultimo modificato
            ckpt_path = max([os.path.join(dir_path, f) for f in candidates], key=os.path.getmtime)
            print(f"[INFO] Checkpoint esatto non trovato, uso il più recente: {ckpt_path}")
    
    try:
        load_model(model, ckpt_path, device)
        print(f"[INFO] Modello caricato da {ckpt_path}")
    except FileNotFoundError:
        print(f"[ERRORE] File checkpoint '{ckpt_path}' non trovato.")
        return

    # 4. Inference
    model.eval()
    correct = 0
    total = 0
    results_per_image = []
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Raccogliamo risultati per immagine (corretto=1, errato=0)
            batch_results = (predicted == labels).cpu().numpy().astype(int)
            results_per_image.extend(batch_results)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"[RESULT] Test Set Accuracy: {accuracy:.2f}%")
    
    plot_test_metrics(results_per_image)
    
    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, classes)

if __name__ == "__main__":
    test(cfg)
