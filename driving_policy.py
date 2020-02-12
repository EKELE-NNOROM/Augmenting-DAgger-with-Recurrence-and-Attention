import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from utils import weights_init_xavierUniform as weights_init
from blocks import Flatten, Recurrent_block, Attention_block, Conv_block_2, Conv_block_3


class OriginalDrivingPolicy(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes


        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(
                in_channels=24,
                out_channels=36,
                kernel_size=4,
                stride=2,
                padding=1),

            nn.ReLU(inplace=False),

            nn.Conv2d(
                in_channels=36,
                out_channels=48,
                kernel_size=4,
                stride=2,
                padding=1),

            nn.ReLU(inplace=False),

            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1),

            nn.ReLU(inplace=False),
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 128, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(64, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(n_classes, n_classes, bias=True),
            nn.ReLU(inplace=False)
        )

        self.apply(weights_init)


    def forward(self, x):
        f = self.features(x)
        logits = self.classifier(f)
        return logits

    def eval(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        return steering_cmd

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)


class RecurrentNetwork(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.CNN1 = Conv_block_2(3, 24, 36, 4, 4, 2, 2, 1, 1)
        self.recurrence1 = Recurrent_block(36, t=4)
        self.recurrence11 = Recurrent_block(36, t=4)
        self.CNN2 = Conv_block_2(36, 48, 64, 4, 4, 2, 2, 1, 1)
        self.recurrence2 = Recurrent_block(64, t=4)
        self.recurrence21 = Recurrent_block(64, t=4)

        self.classifier = nn.Sequential(
            nn.Linear(4096, 128, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(64, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(n_classes, n_classes, bias=True),
            nn.ReLU(inplace=False)
        )

        self.apply(weights_init)


    def forward(self, x):
        f = self.CNN1(x)
        f = self.recurrence1(f)
        f = self.recurrence11(f)
        f = self.CNN2(f)
        f = self.recurrence2(f)
        f = self.recurrence21(f)
        logits = self.classifier(f.view(f.size(0), -1))
        return logits

    def eval(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        return steering_cmd

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)


class AttentionNetwork(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.CNN1 = Conv_block_2(3, 24, 36, 4, 4, 2, 2, 1, 1)
        self.CNN2 = Conv_block_3(36, 36, 36, 36, 3, 3, 3, 1, 1, 1, 1, 1, 1)
        self.att1 = Attention_block(36, 36, 18)
        self.CNN3 = Conv_block_3(36, 36, 36, 36, 3, 3, 3, 1, 1, 1, 1, 1, 1)
        self.CNN4 = Conv_block_2(72, 96, 108, 3, 2, 1, 2, 1, 0)
        self.att2 = Attention_block(108, 108, 54)
        self.CNN5 = Conv_block_3(108, 108, 108, 108, 3, 3, 2, 1, 1, 2, 1, 1, 0)
        self.CNN_branch = Conv_block_2(36, 72, 108, 4, 4, 2, 2, 1, 1)
        self.CNN6 = Conv_block_2(216, 216, 216, 3, 3, 1, 1, 1, 1)

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(3456, 256, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=False),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=False),

            nn.Linear(64, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(n_classes, n_classes, bias=True),
            nn.ReLU(inplace=False)
        )


        self.apply(weights_init)


    def forward(self, x):
        x1 = self.CNN1(x)
        x2 = self.CNN2(x1)
        x3 = self.CNN3(x2)
        xatt1 = self.att1(g=x3,x=x1)
        xatt1 = torch.cat((x3, xatt1), dim=1)
        x4 = self.CNN4(xatt1)
        x5 = self.CNN5(x4)
        x2_cnn = self.CNN_branch(x2)
        xatt2 = self.att2(g=x5, x=x2_cnn)
        xatt2 = torch.cat((x5, xatt2), dim=1)
        x6 = self.CNN6(xatt2)
        x_decoder = self.maxpool(x6)
        flattened = (x_decoder.view(x_decoder.size(0), -1))
        logits = self.classifier(flattened)
        return logits
    def eval(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        return steering_cmd

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)


class RecurrentAttention(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.CNN1 = Conv_block_2(3, 24, 36, 4, 4, 2, 2, 1, 1)
        self.recurrence1 = Recurrent_block(36, t=4)
        self.att1 = Attention_block(36, 36, 18)
        self.recurrence11 = Recurrent_block(36, t=4)
        self.CNN2 = Conv_block_2(72, 96, 108, 4, 4, 2, 2, 1, 1)
        self.att2 = Attention_block(108, 108, 54)
        self.recurrence2 = Recurrent_block(108, t=4)
        self.CNN_branch = Conv_block_2(36, 72, 108, 4, 4, 2, 2, 1, 1)
        self.recurrence21 = Recurrent_block(216, t=4)
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=2),)
        self.classifier = nn.Sequential(
            nn.Linear(3456, 256, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=False),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=False),

            nn.Linear(64, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(n_classes, n_classes, bias=True),
            nn.ReLU(inplace=False)
        )

        self.apply(weights_init)


    def forward(self, x):
        x1 = self.CNN1(x)
        x2 = self.recurrence1(x1)
        #x3 = self.recurrence11(x2)
        x3 = self.recurrence11(x2)
        xatt1 = self.att1(g=x3,x=x1)
        xatt1 = torch.cat((x3, xatt1), dim=1)
        x4 = self.CNN2(xatt1)
        for i in range(4):
            x5 = self.recurrence2(x4)
        # x5 = self.recurrence2(x4)
        x2_cnn = self.CNN_branch(x2)
        xatt2 = self.att2(g=x5, x=x2_cnn)
        xatt2 = torch.cat((x5, xatt2), dim=1)
        x6 = self.recurrence21(xatt2)
        x_decoder = self.maxpool(x6)
        flattened = (x_decoder.view(x_decoder.size(0), -1))
        logits = self.classifier(flattened)
        return logits

    def eval(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        return steering_cmd

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)




class OriginalDrivingPolicyDropOut(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes


        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(
                in_channels=24,
                out_channels=36,
                kernel_size=4,
                stride=2,
                padding=1),

            nn.ReLU(inplace=False),

            nn.Conv2d(
                in_channels=36,
                out_channels=48,
                kernel_size=4,
                stride=2,
                padding=1),

            nn.ReLU(inplace=False),

            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1),

            nn.ReLU(inplace=False),
            Flatten(),
        )

        self.classifier_dropout = nn.Sequential(
            nn.Linear(4096, 128, bias=True),
            nn.ReLU(inplace=False),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes, bias=True),
            nn.ReLU(inplace=False),

            nn.Linear(n_classes, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
        )

        self.apply(weights_init)


    def forward(self, x):
        f = self.features(x)
        logits = self.classifier_dropout(f)
        return logits

    def eval(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        return steering_cmd

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)


class RecurrentNetworkDropOut(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.CNN1 = Conv_block_2(3, 24, 36, 4, 4, 2, 2, 1, 1)
        self.recurrence1 = Recurrent_block(36, t=4)
        self.recurrence11 = Recurrent_block(36, t=4)
        self.CNN2 = Conv_block_2(36, 48, 64, 4, 4, 2, 2, 1, 1)
        self.recurrence2 = Recurrent_block(64, t=4)
        self.recurrence21 = Recurrent_block(64, t=4)

        self.classifier_dropout = nn.Sequential(
            nn.Linear(4096, 128, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.10),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.10),
            nn.Linear(64, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.10),
            nn.Linear(n_classes, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.10),
        )

        self.apply(weights_init)


    def forward(self, x):
        f = self.CNN1(x)
        f = self.recurrence1(f)
        f = self.recurrence11(f)
        f = self.CNN2(f)
        f = self.recurrence2(f)
        f = self.recurrence21(f)
        logits = self.classifier_dropout(f.view(f.size(0), -1))
        return logits

    def eval(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        return steering_cmd

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)


class AttentionNetworkDropOut(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.CNN1 = Conv_block_2(3, 24, 36, 4, 4, 2, 2, 1, 1)
        self.CNN2 = Conv_block_3(36, 36, 36, 36, 3, 3, 3, 1, 1, 1, 1, 1, 1)
        self.att1 = Attention_block(36, 36, 18)
        self.CNN3 = Conv_block_3(36, 36, 36, 36, 3, 3, 3, 1, 1, 1, 1, 1, 1)
        self.CNN4 = Conv_block_2(72, 96, 108, 3, 2, 1, 2, 1, 0)
        self.att2 = Attention_block(108, 108, 54)
        self.CNN5 = Conv_block_3(108, 108, 108, 108, 3, 3, 2, 1, 1, 2, 1, 1, 0)
        self.CNN_branch = Conv_block_2(36, 72, 108, 4, 4, 2, 2, 1, 1)
        self.CNN6 = Conv_block_2(216, 216, 216, 3, 3, 1, 1, 1, 1)

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier_dropout = nn.Sequential(
            nn.Linear(3456, 256, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),

            nn.Linear(64, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(n_classes, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
        )

        self.apply(weights_init)


    def forward(self, x):
        x1 = self.CNN1(x)
        x2 = self.CNN2(x1)
        x3 = self.CNN3(x2)
        xatt1 = self.att1(g=x3,x=x1)
        xatt1 = torch.cat((x3, xatt1), dim=1)
        x4 = self.CNN4(xatt1)
        x5 = self.CNN5(x4)
        x2_cnn = self.CNN_branch(x2)
        xatt2 = self.att2(g=x5, x=x2_cnn)
        xatt2 = torch.cat((x5, xatt2), dim=1)
        x6 = self.CNN6(xatt2)
        x_decoder = self.maxpool(x6)
        flattened = (x_decoder.view(x_decoder.size(0), -1))
        logits = self.classifier_dropout(flattened)
        return logits

    def eval(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        return steering_cmd

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)

class RecurrentAttentionDropOut(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.CNN1 = Conv_block_2(3, 24, 36, 4, 4, 2, 2, 1, 1)
        self.recurrence1 = Recurrent_block(36, t=4)
        self.att1 = Attention_block(36, 36, 18)
        self.recurrence11 = Recurrent_block(36, t=4)
        self.CNN2 = Conv_block_2(72, 96, 108, 4, 4, 2, 2, 1, 1)
        self.att2 = Attention_block(108, 108, 54)
        self.recurrence2 = Recurrent_block(108, t=4)
        self.CNN_branch = Conv_block_2(36, 72, 108, 4, 4, 2, 2, 1, 1)
        self.recurrence21 = Recurrent_block(216, t=4)
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=2),)
        self.classifier_dropout = nn.Sequential(
            nn.Linear(3456, 256, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),

            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),

            nn.Linear(64, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),

            nn.Linear(n_classes, n_classes, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
        )

        self.apply(weights_init)


    def forward(self, x):
        x1 = self.CNN1(x)
        x2 = self.recurrence1(x1)
        #x3 = self.recurrence11(x2)
        x3 = self.recurrence11(x2)
        xatt1 = self.att1(g=x3,x=x1)
        xatt1 = torch.cat((x3, xatt1), dim=1)
        x4 = self.CNN2(xatt1)
        for i in range(4):
            x5 = self.recurrence2(x4)
        # x5 = self.recurrence2(x4)
        x2_cnn = self.CNN_branch(x2)
        xatt2 = self.att2(g=x5, x=x2_cnn)
        xatt2 = torch.cat((x5, xatt2), dim=1)
        x6 = self.recurrence21(xatt2)
        x_decoder = self.maxpool(x6)
        flattened = (x_decoder.view(x_decoder.size(0), -1))
        logits = self.classifier_dropout(flattened)
        return logits

    def eval(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        return steering_cmd

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)
