import torchvision.models as models
import torch
import torchgeo.models
import torch.nn as nn
from dataset_generator import CustomData
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.image as mpimg
import geopandas as gpd

# Hyper-parameters
nb_classes = 15
num_epochs = 2
batch_size = 20
learning_rate = 0.01

device = torch.device('cuda')

model_name = 'resnet50_imagenet'
place = 'manchester'
plot_title = ''
csv_path = 'pop_label_validation2_' + place + '.csv'
validation_dir = 'validation_raster_' + place + '/'

if model_name == 'resnet50_sen2':
    model = torchgeo.models.resnet50(pretrained=False, sensor="sentinel2", bands='all')
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    model.load_state_dict(torch.load('CustomResent_weights_ResNet_sen2_true_bias_10epoch.pth'))


if model_name == 'resnet50_imagenet':
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    model.load_state_dict(torch.load('CustomResent_weights_ResNet_imageNet_true_bias_15epoch.pth'))

if model_name == 'vgg11_imagenet':
    model = models.vgg11()
    model.classifier[6] = nn.Linear(in_features=4096, out_features=9, bias=True)
    model.load_state_dict(torch.load('CustomVGG_weights_ResNet_imageNet_true_bias_15epoch.pth'))

if model_name == 'vgg16_imagenet':
    model = models.vgg16()
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(in_features=4096, out_features=9, bias=True)
    model.load_state_dict(torch.load('CustomVGG16_5epoch.pth'))

model.eval()
model = model.to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.Resize((224, 224)), transforms.Normalize(mean=(0.3268, 0.3520, 0.3334), std=(0.0245, 0.0234, 0.0257))])
dataset = CustomData(csv_file=csv_path, root_dir=validation_dir, transform=transforms)
validation_loader = DataLoader(dataset=dataset, batch_size=batch_size)

pop_list = []
label_list = []
image_path_list = []
prob_list = []

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y, z in loader: # x: images y: 'true' labels z: image file path
            x = x.to(device=device)
            y = y.to(device=device)
            label_list.append(y.tolist())
            scores = model(x)
            for i in list(z):
                image_path_list.append(i)
            sm = torch.nn.Softmax(dim=1)
            top_prob, top_label = torch.topk(sm(scores), 1)

            for i in top_prob.tolist():
                prob_list.append(i[0])

            _, predictions = scores.max(1)
            pop_list.append(predictions.tolist())
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()


check_accuracy(validation_loader, model)
flat_list = [item for sublist in pop_list for item in sublist]
flat_list_label = [item for sublist in label_list for item in sublist]


# Translation


df_category = pd.read_csv("pop_label_a3_testing.csv",usecols =[1, 2, 3], header = None)
df_category = df_category.drop_duplicates()
df_category = df_category.reset_index()
df_category['average'] = (df_category[2] + df_category[3])/2


y = []
y_label = []
for i in flat_list:
    if i == 8:
        y.append(4000)
    else:
        y.append(df_category.iloc[df_category.index[df_category[1]==i]]['average'].item())
for i in flat_list_label:
    if i == 8:
        y_label.append(4000)
    else:
        y_label.append(df_category.iloc[df_category.index[df_category[1]==i]]['average'].item())



# ABS calculation
abs_diff = []
diff = []
for i,j in zip(flat_list, flat_list_label):
    abs_diff.append(abs(i-j))
for i,j in zip(flat_list, flat_list_label):
    diff.append((i-j))

confusion_matrix = confusion_matrix(flat_list_label, flat_list)
df_cm = pd.DataFrame(confusion_matrix)

confidence_prediction = []
sample_frame = pd.DataFrame(list(zip(flat_list, image_path_list, flat_list_label,prob_list)), columns =['prediction', 'image_path', 'true_label', 'probab'])

average_confidence = []
for i in range(9):
    average_confidence.append(sample_frame[sample_frame.prediction == i]['probab'].mean())

category = list(range(9))
values = average_confidence

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(category , values, color='maroon')
plt.xticks(category)
plt.xlabel("Class")
plt.ylim([0, 1])
plt.ylabel("Confidence in %")
plt.title("Prediction Confidence for Each Class, Merseyside")
plt.show()

sum_query_frame = pd.DataFrame()

sample_num = 5
for i in range(9):
    query = 'prediction == ' + str(i)
    query_frame = sample_frame.query(query).sample(sample_num, random_state= 111)
    query_frame = query_frame[query_frame.probab > 0.1]
    sum_query_frame = pd.concat([sum_query_frame, query_frame])

fig = plt.figure()
plt.figure(figsize=(20,10))
for i,j,k,z in zip(range(0, 9*sample_num), sum_query_frame['image_path'], sum_query_frame['true_label'], sum_query_frame['probab']):
    img = mpimg.imread(str(j))
    z = "{:.2f}".format(z)
    # imgplot = plt.imshow(img)
    subtitle = 'true label:' + str(k) +' confidence: ' + str(z)
    plt.subplot(9, sample_num, i+1).set_title(subtitle, x=0.5, y=0.9, color= 'blue')
    imgplot = plt.imshow(img)
    # imgplot = plt.imshow(img, aspect='auto')

    # importing module
    import matplotlib.pyplot as plt

    # assigning x and y coordinates
    y = [0, 1, 2, 3, 4, 5]
    x = [0, 5, 10, 15, 20, 25]

    # depicting the visualization
    plt.plot(x, y, color='green')
    plt.xlabel('x')
    plt.ylabel('y')

    # displaying the title
    plt.axis('off')


fig = plt.figure(figsize=(10, 5))
class_wise_accuracy = list(confusion_matrix.diagonal()/confusion_matrix.sum(axis=0))
plt.bar(category ,class_wise_accuracy)
plt.xticks(category)
plt.xlabel("Class")
plt.ylim([0, 1])
plt.ylabel("Accuracy in %")
plt.title("Class-wise prediction accuracies of Merseyside")
plt.axhline(y=0.1111, color='r', linestyle='-')
plt.show()

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
title = 'Manchester'
model = 'ResNet-50(ImageNet)'
cm_display.ax_.set_title(title + '\'s' + ' ' + 'prediction confusion matrix' + ' ' +'\n Model: ' + model)
cm_display.ax_.set_xlabel('Predicted labels')
cm_display.ax_.set_ylabel('Labels according to the disaggregated census')
plt.show()


residual = []
for i, j in zip(flat_list, flat_list_label):
    residual.append(abs(i-j))

df = pd.read_csv(csv_path, header=None)
df_geometry = pd.DataFrame(list(zip(flat_list, flat_list_label, df[4],residual)), columns =['prediction', 'label', 'geometry', 'residual'])
df_geometry['geometry'] = gpd.GeoSeries.from_wkt(df_geometry['geometry'])

geo_df = gpd.GeoDataFrame(df_geometry, geometry='geometry')
geo_df.to_file('test_leeds.shp', driver='ESRI Shapefile')
ax = geo_df.plot(column="label", legend=True)
ax.set_xlabel('Longitude')
ax.set_ylabel("Latitude")
ax.set_title(title + '\'s' + ' ' + 'categorical labels' + ' ')


ax = geo_df.plot(column="prediction", legend=True)
ax.set_xlabel('Longitude')
ax.set_ylabel("Latitude")
ax.set_title(title + '\'s' + ' ' + 'categorical prediction results' + ' ' +'\n Model: ' + model)
plt.show()

ax = geo_df.plot(column="residual", legend=True)
ax.set_xlabel('Longitude')
ax.set_ylabel("Latitude")
ax.set_title(title + '\'s' + ' ' + 'categorical prediction residuals' + ' ' +'\n Model: ' + model)
plt.show()
