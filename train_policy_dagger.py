import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
from dataset_loader import DrivingDataset
from driving_policy import OriginalDrivingPolicy, RecurrentNetwork, AttentionNetwork, RecurrentAttention
from utils import DEVICE, str2bool
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights",
                        required=True)
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurence",
                        default=True)
    parser.add_argument("--is_l2", type=str2bool, help="whether to do L2 regularization", default=True)
    parser.add_argument("--wd", type=float, help="weight decay rate", default=0.1)
    # parser.add_argument("--policy", type=str, default='OriginalDrivingPolicy',
    #                     help="driving policy of choice e.g. OriginalDrivingPolicy, RecurrentNetwork, AttentionNetwork or RecurrentAttention")
    # parser.add_argument("--output_log", type=str2bool, help="whether to output network logs", default=False)
    return parser


def train_discrete(model, iterator, opt, args):
    model.train()


    loss_hist = []

    params = model.state_dict

    # Do one pass over the data accessed by the training iterator
    # Upload the data in each batch to the GPU (if applicable)
    # Zero the accumulated gradient in the optimizer
    # Compute the cross_entropy loss with and without weights
    # Compute the derivatives of the loss w.r.t. network parameters
    # Take a step in the approximate gradient direction using the optimizer opt
    print("Trainining Full Driving Policy Network...")
    for i_batch, batch in enumerate(iterator):
        x = batch['image']
        y = batch['cmd']

        x = x.to(DEVICE)
        y = y.to(DEVICE)


        if (args.weighted_loss == True):
            class_weights = torch.FloatTensor(args.class_dist).cuda()
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        #
        # YOUR CODE GOES HERE

        pred = model(x)
        opt.zero_grad()
        loss = criterion(pred,y)
        loss.backward()
        opt.step()

        #

        loss = loss.detach().cpu().numpy()
        loss_hist.append(loss)

        PRINT_INTERVAL = int(len(iterator) / 3)
        if (i_batch + 1) % PRINT_INTERVAL == 0:
            print ('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t Time: {:10.3f}'.format(
                i_batch, len(iterator),
                i_batch / len(iterator) * 100,
                np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0),
                time.time() - args.start_time,
            ))

    # I added this
    avg_loss = np.asarray(loss_hist).mean()
    return avg_loss

def accuracy(y_pred, y_true):
    "y_true is (batch_size) and y_pred is (batch_size, K)"
    _, y_max_pred = y_pred.max(1)
    correct = ((y_true == y_max_pred).float()).mean()
    acc = correct * 100
    return acc


def test_discrete(model, iterator, opt, args):
    model.train()

    acc_hist = []

    for i_batch, batch in enumerate(iterator):
        x = batch['image']
        y = batch['cmd']

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        #Y_TRUE.append(y)

        logits = model(x)
        y_pred = F.softmax(logits, 1)
        #Y_PRED.append(y_pred)

        acc = accuracy(y_pred, y)
        acc = acc.detach().cpu().numpy()
        acc_hist.append(acc)
        # I added this


    avg_acc = np.asarray(acc_hist).mean()


    print ('\tVal: \tAcc: {}  Time: {:10.3f}'.format(
        avg_acc,
        time.time() - args.start_time,
    ))

    return avg_acc

def get_class_distribution(iterator, args):
    class_dist = np.zeros((args.n_steering_classes,), dtype=np.float32)
    for i_batch, batch in enumerate(iterator):
        y = batch['cmd'].detach().numpy().astype(np.int32)
        class_dist[y] += 1

    return (class_dist / sum(class_dist))



def main(args,driving_policy=None):
    train_loss = []
    test_acc = []

    data_transform = transforms.Compose([ transforms.ToPILImage(),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                          transforms.RandomRotation(degrees=80),
                                          transforms.ToTensor()])

    training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)

    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=data_transform)

    training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    if (driving_policy == None):
        print("New policy instantiated.")
        # driving_policy = OriginalDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
        # driving_policy = RecurrentNetwork(n_classes=args.n_steering_classes).to(DEVICE)
        driving_policy = AttentionNetwork(n_classes=args.n_steering_classes).to(DEVICE)
        # driving_policy = RecurrentAttention(n_classes=args.n_steering_classes).to(DEVICE)
        args.freeze_last_n_layers = 0

    #opt = torch.optim.Adam(filter(lambda p: p.requires_grad, driving_policy.parameters()), lr=args.lr)
    # Adding weight decay with the weight decay problem fixed with AdamW
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, driving_policy.parameters()), lr=args.lr, weight_decay=0.0005)

    #opt = torch.optim.Adam(filter(lambda p: p.requires_grad, driving_policy.parameters()), lr=args.lr)
    args.start_time = time.time()

    args.class_dist = get_class_distribution(training_iterator, args)

    best_val_accuracy = 0
    for epoch in range(args.n_epochs):
        print ('EPOCH ', epoch)

        curr_train_loss = train_discrete(driving_policy, training_iterator, opt, args)
        train_loss.append(curr_train_loss)

        curr_val_accuracy = test_discrete(driving_policy, validation_iterator, opt, args)

        if (curr_val_accuracy > best_val_accuracy):
            best_val_accuracy = curr_val_accuracy
            torch.save(driving_policy.state_dict(), args.weights_out_file)
            test_acc.append(best_val_accuracy)
        else:
            test_acc.append(curr_val_accuracy)

    return driving_policy, train_loss, test_acc

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    main(args)
