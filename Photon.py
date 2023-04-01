import h5py 
import torch 
import torch.nn as nn 

import torch.optim as optim 
import numpy as np 
import matplotlib.pyplot as plt 
from Split import Datasplit 
import torch.nn.functional as F 
import torchvision 
from tqdm import tqdm
from Stop import EarlyStopping
from sklearn.metrics import accuracy_score
device=torch.device('cuda')

hf1 = h5py.File('Photon.hdf5', 'r')
hf2=h5py.File('Electron.hdf5','r')

print(hf1.keys())
print(hf2.keys())


hf1_x_photon=hf1.get('X')
hf1_y_photon=hf1.get('y')

hf2_x_electron=hf2.get('X')
hf2_y_electron=hf2.get('y')


print(type(hf1_x_photon))
print(type(hf1_y_photon))

#converting to numpy 
numpy_x_photon=np.array(hf1_x_photon)
numpy_y_photon=np.array(hf1_y_photon)

numpy_x_electron=np.array(hf2_x_electron)
numpy_y_electron=np.array(hf2_y_electron)

#converting to tensors 

X_photon=torch.from_numpy(numpy_x_photon)
y_photon=torch.from_numpy(numpy_y_photon)
y_photon=y_photon.view(y_photon.shape[0],1)

X_electron=torch.from_numpy(numpy_x_electron)
y_electron=torch.from_numpy(numpy_y_electron)
y_electron=y_electron.view(y_electron.shape[0],1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(f'photon dataset feature shape {X_photon.shape}')
print(f'photon dataset label shape {y_photon.shape}')

print(f'electron dataset feature shape {X_electron.shape}')
print(f'electron dataset label shape {y_electron.shape}')

print(f'photon label {y_photon[0]}')
print(f'electron label {y_electron[0]}')


X=torch.cat((X_photon,X_electron),0)
X=X.reshape(-1,2,32,32)
print(X.shape)

y=torch.cat((y_photon,y_electron),0)

print(y.shape)

dataset=[]

for i in range(len(X)):
    dataset.append((X[i],y[i]))

#splitint images in train test and validation 

split=Datasplit(dataset,shuffle=True)

train_loader,val_loader,test_loader=split.get_split(batch_size=100,num_workers=8)





# model vgg 19 

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1=nn.Conv2d(2,16,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)

        #size=(16,16,16)

        self.conv3=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.conv4=nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)

        #size=(32,8,8)

        self.conv5=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.conv6=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)

        #size=(64,4,4)
        #fully connected layer 

        self.fc1=nn.Linear(64*4*4,167)
        self.fc2=nn.Linear(167,167)
        self.fc3=nn.Linear(167,2)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=self.pool(x)


        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=self.pool(x)

        x=F.relu(self.conv5(x))
        x=F.relu(self.conv6(x))
        x=self.pool(x)

        x=x.view(-1,64*4*4)

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.sigmoid(self.fc3(x))

        

# calling loss optimizer and model 

model=ConvNet().to(device)

criterion=nn.BCELoss()

optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

num_epochs=50
num_train_loader=len(train_loader)
num_val_loader=len(val_loader)
es=EarlyStopping()
# training+validation loop

for epoch in range(num_epochs):
    train_loss=0.0
    steps = list(enumerate(train_loader))
    pbar = tqdm.tqdm(steps)
    model.train()
    for i,(images,labels) in enumerate(train_loader):
        # origin shape: [100,2,32,32]

        images=images.to(device)
        labels=images.to(device)

        #Forward pass 
        outputs=model(images)
        loss= criterion(outputs,labels)

        #backward and optimizer 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()


        #validation loop 
    model.eval()
    val_loss=0.0
    for i,(data,label) in enumerate(val_loader):

        data=images.to(device)
        label=images.to(device)

        target=model(data)

        loss=criterion(target,label)

        val_loss+=loss.item()

    print(f'Epoch {epoch+1}, Training Loss :{train_loss},Validation Loss :{val_loss}')
    if es(model,val_loss): done = True
    pbar.set_description(f"Epoch: {epoch}, tloss: {train_loss}, vloss: {val_loss:>7f}, EStop:[{es.status}]")
else:
        pbar.set_description(f"Epoch: {epoch}, tloss {train_loss:}")




with torch.no_grad():
    n_correct=0
    n_samples=0

    n_class_correct=[0 for i in range(2)]
    n_class_samples=[0 for i in range(2)]
    for images ,labels in test_loader:
        images = images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted== labels).sum().item()

        for i in range(100):
            label=labels[i]
            pred=predicted[i]
            if (label== pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1
    

    acc=100.0*n_correct/n_samples
    print(f'Accuracy of the network : {acc} %')

    




# prediction+accracy+confusion matrix+ roc_auc curve 