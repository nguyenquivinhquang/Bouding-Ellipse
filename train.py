from torch.utils.data import DataLoader
from dataset.Custom_image_dataset import CustomImageDataset
from numpy.lib.npyio import save
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
from models.resnet import fbresnet18
from models.loss_function import ellipse_loss
import argparse
import os
# ----- Init param -----#
parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-google_drive', type=str, default='./', help='Train on colab')
parser.add_argument('-total_epoch', type=int, default=200, help='Total epoch want to train')

args = parser.parse_args()

# ----- Parameter ---- #
batch_size = args.batch_size    
dataset_download = True
learning_rate = 0.1
path = args.google_drive
total_epoch = args.total_epoch

label_path = 'dataset/train_label.csv'
img_path = 'dataset/img/'
try: os.mkdir('.checkppoint')
except: pass


batch_size = 2
total_epoch = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#-------handle data ------- #
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Resize((224,224), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = transforms.Compose(transform_train_list)
    
dataset  = CustomImageDataset(img_dir=img_path, label_dir=label_path, transform=data_transforms)
total_data = len(dataset)
total_train = int(0.8 * total_data)
total_val = total_data - total_train
train_set, val_set = torch.utils.data.random_split(dataset, [total_train, total_val])
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

#------ Define network ----- #
model = fbresnet18()
model.train()
model.to(device)

#---------Define loss function ------------#
criterion = ellipse_loss()
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Define loss history
loss_history = []
cur_epoch = 0

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
       
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        print(outputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss /= train_loss/(batch_idx+1)
    print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss, 100.*correct/total, correct, total))
    save_path = path + "checkpoint/Cifar10-epoch-" + str(epoch) + ".pth"
    
    # Save model afer each epoch
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,}, save_path )
    
    
    loss_history.append(train_loss)

if __name__ == '__main__':    
    for epoch in range(cur_epoch, total_epoch):
        
        time_start = time.time()
        train(epoch)
        # validate()
        time_elapsed = time.time() - time_start
        print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
        if (epoch // 5 == 0): scheduler.step()
        print('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))