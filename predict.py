import torch
from PIL.Image import Image
from torchvision import models
from torch import optim
import numpy as np
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', type=int, dest='top_k', default=5)
    parser.add_argument('--category_names', type=str, dest="category_names")
    parser.add_argument('--gpu', type=bool, dest="gpu", default=True)

    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    with Image.open(image) as pil_image:
        means = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # load the image
        #         pil_image = Image.open(image)
        w, h = pil_image.size

        # resize the image and maintain the aspect ratio
        if w > h:
            pil_image.thumbnail((1000, 256))
        else:
            pil_image.thumbnail((256, 1000))

        # update w and h to match the new width and h
        w, h = pil_image.size

        l = (w - 224) / 2
        t = (h - 224) / 2
        r = (w + 224) / 2
        b = (h + 224) / 2

        pil_image = pil_image.crop(box=(l, t, r, b))


        # convert the image to an ndarray
        np_image = np.array(pil_image)
        # normalize the color channels to floats between 0 and 1
        np_image = np_image / 255
        np_image = (np_image - means) / std

        # rearranging the channels so they match what the pytorch models expect
        np_image = np_image.transpose((2, 0, 1))

        torch_image = torch.from_numpy(np_image)

    return torch_image

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    #set up the model based on the arch type it was made from
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
    elif checkpoint['arch'] == 'densenet201':
        model = models.densene201(weights='DEFAULT')

    model.load_state_dict(checkpoint['state_dict'])
    class_to_idx = checkpoint['class_to_idx']

    return model, class_to_idx

def predict():
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

    #get the arguments
    args = get_args()

    # set the device
    device = ("cuda" if args.gpu else "cpu")

    # open file with category info
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    # load checkpoint
    model, class_to_idx = load_checkpoint(filepath=args.checkpoint)
    model.to(device)
    image = process_image(args.image_path)
    image.to(device)
    inverted_class_to_idx = {value: key for key, value in class_to_idx.items()}

    with torch.no_grad():
        logps = model(image)

    ps = torch.exp(logps)

    probs, classes = ps.topk(args.top_k)

    new_classes = list()

    for c in classes:
        new_classes.append(inverted_class_to_idx[c])

    class_labels = [cat_to_name[cat_idx] for cat_idx in new_classes]

    for i in range(args.top_k):
        print(f"Flower N *ame: {class_labels[i]} Probability: {probs.tolist()[i]}")

if __name__ == '__main__':
    predict()
