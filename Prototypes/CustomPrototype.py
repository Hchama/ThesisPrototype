import torch
from torchvision import transforms
from PIL import Image
import fastai
import timm
import torch.nn as nn
import matplotlib.pyplot as plt

#model architectures
model1 = timm.create_model('resnet50', pretrained=True)
model2 = timm.create_model('efficientnet_b0', pretrained=True)

#classifiers used
class CustomModel(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.classifier = nn.Sequential(
            nn.Linear(1000 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x1 = self.model1(x.clone())
        x2 = self.model2(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

#img input
image_path = input("Enter the path to the image: ")

# Loading the trained weights
model = CustomModel(model1, model2, num_classes=2)
model.load_state_dict(torch.load('models/custom_model.pth', map_location=torch.device('cpu')))
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

#-----------------making the Classification-------------#
output = model(image)
_, predicted_index = torch.max(output, 1)
predicted_class_name = index_to_class_name[predicted_index.item()]

#-------------------------display---------------------------#
plt.imshow(original_image)
plt.axis('off')
plt.title(f'Predicted class: {predicted_class_name}')
plt.show()