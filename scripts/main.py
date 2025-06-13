import torch.nn as nn
import torch
from torch.utils.data import DataLoader,random_split

from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from loader import loadTrainVal,loadTest

from model import PneumoniaClassifier

from train import train

from evaluate import eval


if __name__=="__main__":

    #Define data directory and train transformations
    data = r'../Data/chest_xray/chest_xray'
    transforms=tt.Compose([tt.Resize(255),
                                            tt.CenterCrop(224),
                                            tt.RandomHorizontalFlip(),
                                            tt.RandomRotation(10),
                                            tt.RandomGrayscale(),
                                            tt.RandomAffine(translate=(0.05,0.05), degrees=0),
                                            tt.ToTensor()

                                           ])
    
    #Create a dataset
    
    dataset = ImageFolder(data+'/train', transform=transforms)

    #Split the data

    trainSize=round(0.71*len(dataset))
    valSize=len(dataset)-trainSize
    train, val = random_split(dataset, [trainSize, valSize])

    #Create DataLoaders

    trainLoader,valLoader=loadTrainVal(data,transforms,16,2)

    #define device

    device="cuda" if torch.cuda.is_available() else "cpu"

    #Instantiate model

    model=PneumoniaClassifier().to(device)

    #Train the model

    classifier=train(model,trainLoader,valLoader,device,10,0.001)

    #Define test transformations
    testTransforms=tt.Compose([tt.Resize(255),
                                            tt.CenterCrop(224),
                                            tt.ToTensor()
                                           ])
    
    #Load the test DataLoader
    testLoader=loadTest(data,testTransforms,16)

    #Evaluate the classifier

    eval(classifier,testLoader,device)





    #Save the model
    torch.save(classifier.state_dict(), 'TrainedModel.pth')


    






    




