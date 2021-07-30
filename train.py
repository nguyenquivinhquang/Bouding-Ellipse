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
from models.resnet import fbresnet18,fbresnet50
from models.loss_function import ellipse_loss, get_IOU_loss, compute_diff_angle
import argparse
import os

# ----- Init param -----#
parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-google_drive', type=str, default='./', help='Train on colab')
parser.add_argument('-total_epoch', type=int, default=200, help='Total epoch want to train')
parser.add_argument('-save_model', type=str, default='./', help='Path to save trained model')
parser.add_argument('-num_workers',type=int,default=4,help='The number of threads')
parser.add_argument('-threshold',type=int,default=0.1,help='The threshold for accepted differece area and angle')
parser.add_argument('-resume',type=str,default=None,help='Continue to train model, add the model weigth want to continue to train')

args = parser.parse_args()

# ----- Parameter ---- #
batch_size = args.batch_size    
dataset_download = True
learning_rate = 0.1
path = args.google_drive
total_epoch = args.total_epoch
save_model = args.save_model
num_workers = args.num_workers
thresh = args.threshold
label_path = path + "/train_label.csv"
img_path = path + "/img/"

print(label_path)



# batch_size = 2
# total_epoch = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#-------handle data ------- #
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = transforms.Compose(transform_train_list)
    
dataset  = CustomImageDataset(img_dir=img_path, label_dir=label_path, transform=data_transforms)
total_data = len(dataset)
total_train = int(0.9 * total_data)
total_val = total_data - total_train
train_set, val_set = torch.utils.data.random_split(dataset, [total_train, total_val])
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#------ Define network ----- #
model = fbresnet50()
model.train()
model.to(device)

# Define loss history
loss_history = []
best_loss = 1000000
cur_epoch = 0

#---------Define loss function ------------#
criterion = ellipse_loss()
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#---- Resume trainning
if args.resume:
    weigth_path = args.resume
    checkpoint = torch.load(weigth_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # scheduler.load_state_dict(checkpoint['optimizer_state_dict'])
    cur_epoch = checkpoint['epoch']
    print("Resume trainning at epoch:", cur_epoch)

def train(epoch):
    global best_loss
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
       
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, targets)
       
        loss.backward()
        optimizer.step()
        # print(outputs.shape)
        train_loss += loss.item()
        total += targets.size(0)
        iou = get_IOU_loss(outputs,targets)
        cond1, cond2 = iou < thresh,compute_diff_angle(outputs, targets) < thresh

        # print(cond1.shape, cond2.shape)

        correct += torch.sum(iou[cond1|cond2 ]) # |: bitwise or
    print(train_loss, batch_idx)
    train_loss = train_loss / (batch_idx+1)
    print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss, 100.*correct/total, correct, total))
    if epoch//2 == 0 or train_loss < best_loss:
        print("saving model at", epoch)
        save_path = save_model + "/checkpoint/ellipse-epoch-" + str(epoch) + ".pth"
        if train_loss < best_loss:
            save_path = save_model + "/checkpoint/ellipse-best-epoch-" + str(epoch) + ".pth"
            best_loss = train_loss
        # Save model afer each epoch
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,}, save_path )
        
    loss_history.append(train_loss)
def validate():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            print(outputs[0])
            exit
            loss = criterion(outputs, targets).mean()
            test_loss += loss.item()
            total += targets.size(0)
            iou = get_IOU_loss(outputs,targets)
            correct += torch.sum(iou[iou < 0.1])

    test_loss /= test_loss/(batch_idx+1)
    print(len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss, 100.*correct/total, correct, total))


if __name__ == '__main__':    
    for epoch in range(cur_epoch, total_epoch):
        time_start = time.time()
        train(epoch)
        
        time_elapsed = time.time() - time_start
        
        print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
        
        validate()

        scheduler.step()
        print('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))