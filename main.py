# imports
import tkinter as tk
from tkinter import filedialog, ttk
import torch
import timm
from torchvision import transforms
from PIL import Image
from fastai import *
from fastai.vision.all import *
import torch.nn as nn
import matplotlib.pyplot as plt

# Custom model classifiers
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

# Other model classifiers
num_classes = 2 
classifier_head = nn.Sequential(
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

#-----------------Functions------------------#

def get_model(model_choice):
    if model_choice == 'resnet50':
        model_arch = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model_weight = 'FinalModels/resnet50_model.pth'
        print ('You picked resnet50')
    elif model_choice == 'resnet34':
        model_arch = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        model_weight = 'FinalModels/resnet34_model.pth'
        print ('You picked resnet34')
    elif model_choice == 'vit':
        model_arch = models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        model_weight = 'FinalModels/vit_model.pth'
        print ('You picked vit')
    elif model_choice == 'vgg':
        model_arch = models.vgg19(weights=VGG19_Weights.DEFAULT)
        model_weight = 'FinalModels/vgg19_model.pth'
        print ('You picked vgg')
    elif model_choice == 'Custom Model':
        model1 = timm.create_model('resnet50', pretrained=True)
        model2 = timm.create_model('efficientnet_b0', pretrained=True)
        model_arch = CustomModel(model1, model2, num_classes=2)
        model_weight = 'FinalModels/custom_model.pth'
        print ('You picked the custom model')
    else:
        raise ValueError(f'Unknown model: {model_choice}')
    
    return model_arch, model_weight

def classify_image():
    # Open file dialog to select image
    image_path = filedialog.askopenfilename()

    model_arch ,model_weight = get_model(model_var.get())
    
    # picking the appropriate classifiers for the model
    if model_var.get() != 'Custom Model':
        model = nn.Sequential(model_arch, classifier_head)
        model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')), strict=False)
    else:
        model = model_arch
        model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')))

    model.eval()

    # Transforming the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    original_image = Image.open(image_path).convert('RGB')
    image = transform(original_image)
    image = image.unsqueeze(0)

    # Classification
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

# Creates the window UI
window = tk.Tk()

# Set window size
window_width = 300
window_height = 200

# Get screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calculate position coordinates to display in the middle of the screen
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

# Set the window's size and position
window.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# Add a dropdown menu to select the model
model_var = tk.StringVar(window)
model_var.set('resnet50')  # default option
model_choices = {'resnet50', 'resnet34', 'Custom Model', 'vit', 'vgg'}
model_popupMenu = ttk.Combobox(window, textvariable=model_var, values=list(model_choices))
model_popupMenu.pack()

# Add a button to select an image and classify it
classify_button = tk.Button(window, text="Pick an image to classify", command=classify_image)
classify_button.pack()

# Start the Tkinter event loop
window.mainloop()