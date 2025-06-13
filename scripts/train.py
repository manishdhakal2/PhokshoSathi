import torch.nn as nn
import torch
from torch.utils.data import DataLoader




def train(classifier:nn.Module, trainLoader:DataLoader, valLoader:DataLoader, device, epochs=10, lr=0.001):
    """
    Train the model on the training data and validate it on the validation data.
    
    Parameters:
    - model: The model to be trained.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - device: Device to run the model on (CPU or GPU).
    - epochs: Number of epochs to train the model.
    - lr: Learning rate for the optimizer.
    """

    #Define the loss function
    
    loss_fn = nn.CrossEntropyLoss()

    #Define the optimizer

    epochs=10
    lr=0.0001
    optimizer=torch.optim.Adam(classifier.parameters(),lr=lr)
    
    #Loop through the epochs
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0

        # Loop through the training data
        for batch in trainLoader:
            images, labels = batch[0].to(device), batch[1].to(device)
            # Forward pass
            y_pred = classifier(images)
            loss = loss_fn(y_pred, labels)

            # Backward pass and optimization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # Print the loss for the epoch
        print(f"\nEpoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(trainLoader):.4f}")
        print(f"GPU Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        # Validate the model
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in valLoader:
                images, labels = batch[0].to(device), batch[1].to(device)
        
                outputs = classifier(images) 
                loss = loss_fn(outputs, labels)  
                val_loss += loss.item()
        
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)


        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss/len(valLoader)}, Accuracy: {val_accuracy}%")
    
    return classifier




