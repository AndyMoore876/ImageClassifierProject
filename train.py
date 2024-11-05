import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', dest='save_dir')
    parser.add_argument('--arch', type=str, default='vgg16', dest='arch')
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.03)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, dest='epochs', default=10)
    parser.add_argument('--gpu', type=bool, dest='gpu', default='gpu')

    return parser.parse_args()
def train():

    #get command line arguments

    args = get_args()

    # set directories
    train_dir = args.data_directory + '/train'
    valid_dir = args.data_directory + '/valid'
    test_dir = args.data_directory + '/test'

    # set up image transforms and loaders
    training_transform = transforms.Compose([transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    validation_transform = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    training_images = datasets.ImageFolder(train_dir, transform=training_transform)
    test_images = datasets.ImageFolder(test_dir, transform=testing_transform)
    validation_images = datasets.ImageFolder(test_dir, transform=validation_transform)

    trainloader = torch.utils.data.DataLoader(training_images, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_images, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_images, batch_size=64)


    # set the device
    device = ("cuda" if args.gpu == "gpu" else "cpu")
    model = None
    #set up the model architectures
    if args.arch == 'vgg16':
        # set up the model based on vgg16 weights
        model = models.vgg16(weights='DEFAULT')

        # freezes the model parameters as they are
        for para in model.parameters():
            para.requires_grad = False

        # definition of our classifier
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, args.hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.3)),
            ('fc2', nn.Linear(args.hidden_units, 256)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(256, 102)),
            ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == 'densenet201':
        # set up the model based on vgg16 weights
        model = models.densene201(weights='DEFAULT')

        # freezes the model parameters as they are
        for para in model.parameters():
            para.requires_grad = False

        # definition of our classifier
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1920, args.hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.3)),
            ('fc2', nn.Linear(args.hidden_units, 256)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(256, 102)),
            ('output', nn.LogSoftmax(dim=1))]))

    # updating the model with our classifier
    model.classifier = classifier

    criterion = nn.NLLLoss()

    # training only the model classifier parameters as the other parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model.to(device)
    criterion.to(device)
    current_epoch = 0
    epochs = args.epochs
    running_loss = 0
    print_every = 10
    steps = 0
    valid_loss = 0

    train_losses = []
    validation_losses = []

    for epoch in range(epochs):
        current_epoch = epoch
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs).to(device)

            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        # inputs = inputs.view(inputs.shape[0],768)
                        logps = model.forward(inputs)
                        loss = criterion(logps.to(device), labels.to(device))

                        valid_loss += loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / len(trainloader))
                validation_losses.append(valid_loss / len(validationloader))

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Device {device} "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {valid_loss / len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(validationloader)*100:.3f}")
                running_loss = 0
                valid_loss = 0
                model.train()
    save_checkpoint(args.arch, model, current_epoch, optimizer, training_images, args.save_directory)


def save_checkpoint(arch,model, epoch, optim, training_images, save_dir):
    # TODO: Save the checkpoint

    checkpoint = {'input_size': 768,
                  'output_size': 102,
                  # 'hidden_layers': [hidden_layer.out_features for hidden_layer in model.classifier],
                  'classifier': model.classifier.state_dict(),
                  'current_epoch': epoch,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': training_images.class_to_idx,
                  'arch':arch
                  }

    torch.save(checkpoint, save_dir+'checkpoint.pth')

if __name__ == "__main__":
    train()
