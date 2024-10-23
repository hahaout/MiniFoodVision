import torch as T
import torch.nn as nn
# a function to create conv_2d
def create_conv2d(hidden_units):
    layer = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
    )
    return layer

## Create a model by subclassing nn.module
import torch as T
import torch.nn as nn


class VGGV0(nn.Module):
    def __init__(self, input_size,
                  hidden_units,
                    output_size):
        super().__init__()
        self.Conv2d_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.Conv2d_layer_2 = create_conv2d(hidden_units=hidden_units)
        self.Conv2d_layer_3 = create_conv2d(hidden_units=hidden_units)
        self.Conv2d_layer_4 = create_conv2d(hidden_units=hidden_units)
        self.Conv2d_layer_5 = create_conv2d(hidden_units=hidden_units)
        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*49,
                      out_features=output_size)
        )

    def forward(self, x):
        #x = self.Conv2d_layer_1(x)
        #print(x.shape)
        #x = self.Conv2d_layer_2(x)
        #print(x.shape)
        #x = self.Conv2d_layer_3(x)
        #print(x.shape)
        #x = self.Conv2d_layer_4(x)
        #print(x.shape)
        #x = self.Conv2d_layer_5(x)
        #print(x.shape)
        #x = self.Classifier(x)
        #print(x.shape)
        return self.Classifier(self.Conv2d_layer_5(self.Conv2d_layer_4(\
            self.Conv2d_layer_3(self.Conv2d_layer_2(self.Conv2d_layer_1(x)))))) 