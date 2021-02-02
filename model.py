from jittor.models import Resnet50
import jittor.nn as nn 

class Net(nn.Module):
    def __init__(self, num_classes):
        self.base_net = Resnet50(num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        return x 