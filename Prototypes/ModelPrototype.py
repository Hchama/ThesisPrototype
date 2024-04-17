import torch
from torchvision import transforms
from PIL import Image
from fastai import *
from fastai.vision.all import *
import torch.nn as nn
import matplotlib.pyplot as plt

#Get model weight and architecture function
def get_model():
    model_choice = input("Choose a model (e.g., 'resnet50', 'resnet34', 'densenet', 'vit', 'vgg'): ")
    if model_choice == 'resnet50':
        model_arch = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model_weight = 'models/resnet50_model.pth'
    elif model_choice == 'resnet34':
        model_arch = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        model_weight = 'models/resnet34_model.pth'
    elif model_choice == 'densenet':
        model_arch = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        model_weight = 'models/densenet_model.pth'
    elif model_choice == 'vit':
        model_arch = models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        model_weight = 'models/vit_model.pth'
    elif model_choice == 'vgg':
        model_arch = models.vgg11(weights=VGG11_BN_Weights.DEFAULT)
        model_weight = 'models/vgg_model.pth'
    else:
        raise ValueError(f'Unknown model: {model_choice}')
    
    return model_arch, model_weight

#classifiers used
num_classes = 2 
classifier_head = nn.Sequential(
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

model_arch ,model_weight = get_model()
#img input
image_path = input("Enter the path to the image: ")

# Load trained model weights and adjust them
state_dict = torch.load(model_weight, map_location=torch.device('cpu'))
state_dict['1.4.weight'] = classifier_head[3].weight
state_dict['1.4.bias'] = classifier_head[3].bias

model = nn.Sequential(model_arch, classifier_head)
model.eval()

#Transforming the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

original_image = Image.open(image_path).convert('RGB')
image = transform(original_image)
image = image.unsqueeze(0)

#Classification
class_names = ['AI', 'Authentic']
index_to_class_name = {i: name for i, name in enumerate(class_names)}

#-----------------making the prediction-------------#
output = model(image)
_, predicted_index = torch.max(output, 1)
predicted_class_name = index_to_class_name[predicted_index.item()]

#-------------------------display---------------------------#
plt.imshow(original_image)
plt.axis('off')
plt.title(f'Predicted class: {predicted_class_name}')
plt.show()