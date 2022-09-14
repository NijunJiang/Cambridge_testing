import torchgeo.models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from dataset_generator import CustomData
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import gc
import sys
import torchvision.models as models
from torchvision.models import ResNet18_Weights, resnet50, ResNet50_Weights, vgg16, VGG16_Weights, vgg11, VGG11_Weights
# device

device = torch.device('cuda')
torch.manual_seed(1453)
# hyper-parameters for deep learning
nb_classes = 15
num_epochs = 15
batch_size = 32
learning_rate = 0.01

model_name = 'vgg16_imagenet'
place = 'southengland'

csv_path = 'pop_label_validation2_' + place + '.csv'
csv_val = 'pop_label_ee3_testing.csv'
dir_file = 'validation_raster_' + place + '/'
# transformer transforms.Normalize(mean=(0.3485, 0.3596, 0.3203), std=(0.1497, 0.1432, 0.1411))
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=(0.3268, 0.3520, 0.3334), std=(0.0245, 0.0234, 0.0257))])

if model_name == 'resnet50_sen2':
    model = torchgeo.models.resnet50(pretrained=True, sensor="sentinel2", bands='all')
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)

if model_name == 'resnet50_imagenet':
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)

if model_name == 'vgg11_imagenet':
    model = models.vgg11(weights = VGG11_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(in_features=4096, out_features=9, bias=True)
    # model.classifier[0].requires_grad_(True)
    # model.classifier[3].requires_grad_(True)

if model_name == 'vgg16_imagenet':
    model = models.vgg16(weights = VGG16_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(in_features=4096, out_features=9, bias=True)
    # model.classifier[0].requires_grad_(True)
    # model.classifier[1].requires_grad_(True)
    # model.classifier[2].requires_grad_(True)
    # model.classifier[3].requires_grad_(True)
    # model.classifier[4].requires_grad_(True)
    # model.classifier[5].requires_grad_(True)
print(model)

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0
    for images,_,_ in loader:
        channels_sum += torch.mean(images, dim = [0,2,3])
        channels_squared_sum += torch.mean(images**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches-mean**2)*0.5

    return  mean, std


# dataset loading
dataset = CustomData(csv_file=csv_path, root_dir=dir_file, transform=transforms)

# dataset_testing = CustomData(csv_file='pop_label_a3_testing.csv', root_dir='city_raster2', transform=transforms)
dataset_testing = CustomData(csv_file=csv_val, root_dir='google_ee_raster', transform=transforms)
print('dataset loaded')
# sampler
df = pd.read_csv(csv_path, header=None)
weight_list = df[1].value_counts()
weight_list = weight_list.to_frame()
class_weights = (1/weight_list[1]).to_frame()
df = pd.read_csv(csv_path, header=None)
sample_weights = []

for i in df[1].tolist():
    class_weight = class_weights.loc[[i]].values[0][0]
    sample_weights.append(class_weight)


sampler_training = WeightedRandomSampler(sample_weights, len(dataset), replacement=True)

df = pd.read_csv(csv_val, header=None)
weight_list = df[1].value_counts()
weight_list = weight_list.to_frame()
class_weights = (1/weight_list[1]).to_frame()
df = pd.read_csv(csv_val, header=None)
sample_weights = []
print('csv loaded')
for i in df[1].tolist():
    class_weight = class_weights.loc[[i]].values[0][0]
    sample_weights.append(class_weight)

'''
# Translation table

df_category = pd.read_csv("pop_label_a3_testing.csv",usecols =[1, 2, 3], header = None)
df_category = df_category.drop_duplicates()
df_category = df_category.reset_index()
df_category['average'] = (df_category[2] + df_category[3])/2


x = [1, 8, 0, 1, 2]
y = []
print(df_category[1]==1)
print(df_category.index[df_category[1]==1])
for i in x:
    y.append(df_category.iloc[df_category.index[df_category[1]==i]]['average'].item())


print(y, sum(y))

'''




sampler_testing = WeightedRandomSampler(sample_weights, len(dataset))
print('before')
# data loader
dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler_training)
dataset_loader_testing = DataLoader(dataset=dataset_testing, batch_size=batch_size)
# mean, std = get_mean_std(dataset_loader)
# print(mean,std)
# model freezing for resnet
if model_name == 'resnet50_imagenet' or model_name == 'resnet50_sen2':
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 4:
            for param in child.parameters():
                param.requires_grad = False
if model_name == 'resnet50_sen2':
    model.conv1.requires_grad_(True)

model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)

# Training pipeline
n_total_steps = len(dataset_loader)
print('start training')
for epoch in range(num_epochs):
    losses = []
    for i, (images, labels, _) in enumerate(dataset_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, prediction = outputs.max(1)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        print(labels)
        print(prediction)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
    mean_loss = sum(losses) / len(losses)

print('Finished Training')

pop_list = []
label_list = []
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            label_list.append(y.tolist())
            scores = model(x)
            _, predictions = scores.max(1)
            pop_list.append(predictions.tolist())
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            print(predictions)
            print(y)
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()


check_accuracy(dataset_loader_testing, model)
File = 'CustomVGG16_15epoch.pth'
torch.save(model.state_dict(), File)
print('model parameters saved')
flat_list = [item for sublist in pop_list for item in sublist]
flat_list_label = [item for sublist in label_list for item in sublist]
confusion_matrix = confusion_matrix(flat_list, flat_list_label)
print(flat_list)
df_cm = pd.DataFrame(confusion_matrix)
plt.figure(figsize = (10, 7))
sn.heatmap(df_cm)
plt.show()


# Translation

df_category = pd.read_csv(csv_val, usecols =[1, 2, 3], header = None)
df_category = df_category.drop_duplicates()
df_category = df_category.reset_index()
df_category['average'] = (df_category[2] + df_category[3])/2
print(df_category)

y = []

for i in flat_list:
    y.append(df_category.iloc[df_category.index[df_category[1]==i]][2].item())

print(y, sum(y))


# clear cache
model.cpu()
del model
gc.collect()
torch.cuda.empty_cache()
