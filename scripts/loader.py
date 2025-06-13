from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split



def loadTrainVal(filepath:str,transforms,batch_size:int,num_workers:int)->tuple:
    """
    Returns a tuple of 2 DataLoader objects(train,val) (randomly splitted) created from the data from the filepath

    params:
    filepath:Path to the file location
    transforms:Transformations to be applied
    batch_size: The Batch size of each loader
    """
    #Create a dataset of images from the given filepath
    dataset = ImageFolder(filepath+'/train', transform=transforms)


    trainSize=round(0.71*len(dataset))
    valSize=len(dataset)-trainSize

    #Split the data into training and validation
    train, val = random_split(dataset, [trainSize, valSize])

    
    trainLoader = DataLoader(train, batch_size, shuffle=True, num_workers=2)
    valLoader = DataLoader(val, batch_size*2, num_workers=2)

    return trainLoader, valLoader


def loadTest(filepath:str,transforms,batch_size:int)->DataLoader:

    dataset = ImageFolder(filepath+'/test', transform=transforms)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
