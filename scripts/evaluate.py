import torch.nn as nn
import torch


def eval(classifier, testLoader:torch.utils.data.DataLoader, device:torch.device):
    """
    Evaluate the model on the given dataloader.

    Parameters:
    classifer : The model to evaluate.
    testLoader (DataLoader): The dataloader containing the data to evaluate.
    device (torch.device): The device to run the evaluation on.

    Returns:
    float: The average loss over the dataset.
    float: The accuracy of the model on the dataset.
    """
    
    classifier.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in testLoader:
            images, labels = batch[0].to(device), batch[1].to(device)

            outputs = classifier(images) 
            loss = loss_fn(outputs, labels)  
            test_loss += loss.item()

            values, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss/len(testLoader)}, Accuracy: {test_accuracy}%")