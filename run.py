# PlacesCNN for scene classification

import argparse

import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.autograd import Variable as V
from torch.nn import functional as F
from PIL import Image

parser = argparse.ArgumentParser(description='Run Scene Classifier with Pretrained Weights')
parser.add_argument('--arch', default='resnet18')
parser.add_argument('data', metavar='IMG', help='path to image file')
parser.add_argument('--num_classes', type=int, help='num of classes in the model')

def main():
    global args
    args = parser.parse_args()
    print(args)
    
    model_file = '%s_best.pth.tar' % args.arch
        
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    
    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load the class labels
    file_name = 'subset_data_classes.txt'

    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0])
    classes = tuple(classes)
    print (classes)
        
    # load the test image
    img_name = args.data
    img = Image.open(img_name).convert('RGB')
    input_img = V(centre_crop(img).unsqueeze(0))
    
    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # print the output
    print('{} prediction on {}'.format(args.arch,img_name))
    # output the prediction
    for i in range(0, args.num_classes):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
    
    

if __name__ == '__main__':
    main()
