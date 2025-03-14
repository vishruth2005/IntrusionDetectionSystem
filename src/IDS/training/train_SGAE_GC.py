import torch
import logging
import torch.nn as nn
from tqdm import tqdm

def train_scae_gc_model(scae_gc_model, train_loader, num_epochs, learning_rate, device):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    scae_gc_model.to(device)
    
    logging.info("SGAE_GC training started successfully.")
    
    for cae in [scae_gc_model.cae1, scae_gc_model.cae2, scae_gc_model.cae3]:
        for param in cae.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, scae_gc_model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        scae_gc_model.train()
        running_loss = 0.0  
        correct = 0  
        total = 0  

        try:
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):  
                inputs, labels = batch  
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  

                optimizer.zero_grad()
                outputs = scae_gc_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()  
                _, predicted = outputs.max(1)
                total += labels.size(0)  
                correct += predicted.eq(labels).sum().item()  
        except Exception as e:
            logging.error(f"Error in epoch {epoch + 1}: {str(e)}")
            continue
        
    logging.info("Training completed")  
    return scae_gc_model