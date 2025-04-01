import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure targets are in the same shape as inputs
        targets = targets.view(-1, 1)  # Reshape targets if necessary
        inputs = inputs.view(-1, inputs.size(-1))  # Flatten inputs

        # Calculate the softmax probabilities
        probs = torch.softmax(inputs, dim=-1)
        
        # Get the log probabilities
        log_probs = torch.log(probs + 1e-7)  # Add epsilon to avoid log(0)

        # Gather the probabilities of the true classes
        true_class_probs = probs.gather(1, targets)

        # Calculate the focal loss components
        focal_loss = -self.alpha * (1 - true_class_probs) ** self.gamma * log_probs.gather(1, targets)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_scae_gc_model(scae_gc_model, train_loader, num_epochs, learning_rate, device):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    scae_gc_model.to(device)
    
    logging.info("SGAE_GC training started successfully.")
    
    for cae in [scae_gc_model.cae1, scae_gc_model.cae2, scae_gc_model.cae3]:
        for param in cae.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, scae_gc_model.parameters()), lr=learning_rate)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)  # Use custom Focal Loss

    for epoch in range(num_epochs):
        scae_gc_model.train()
        running_loss = 0.0  
        correct = 0  
        total = 0  
        all_labels = []
        all_predictions = []

        try:
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):  
                inputs, labels = batch  
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  

                optimizer.zero_grad()
                outputs = scae_gc_model(inputs)
                loss = criterion(outputs, labels)  # Use custom Focal Loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()  
                _, predicted = outputs.max(1)
                total += labels.size(0)  
                correct += predicted.eq(labels).sum().item()  
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        except Exception as e:
            logging.error(f"Error in epoch {epoch + 1}: {str(e)}")
            continue
        
        # Calculate precision and recall
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss / len(train_loader):.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
    logging.info("Training completed")  
    return scae_gc_model