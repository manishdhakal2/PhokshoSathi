import torch.nn as nn
from torchvision.models import resnet50,ResNet50_Weights

class PneumoniaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        #Load the PreTrained Resnet50
        self.model=resnet50(weights=ResNet50_Weights.DEFAULT)
        
        for params in self.model.parameters():
            #Freeze all parameters
            params.require_grad=False

        #Create a new fully connected layer with unfrozen params
        self.model.fc=nn.Linear(self.model.fc.in_features,2)

    def forward(self,x):
        return self.model(x)
    