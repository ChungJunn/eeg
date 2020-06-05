import torch
import torch.nn as nn

class EEG_AE_MODEL(nn.Module):
    def __init__(self, dim_input=1, dim_layer=2, dim_z=3):
        super(EEG_AE_MODEL, self).__init__()

        self.enc1 = nn.Linear(dim_input, dim_layer)
        self.enc2 = nn.Linear(dim_layer, dim_z)

        self.dec1 = nn.Linear(dim_z, dim_layer)
        self.dec2 = nn.Linear(dim_layer, dim_input)

    def forward(self, input):
        z = self.enc2(self.enc1(input))
        output = self.dec2(self.dec1(z))
        return output

if __name__ == '__main__':
    import numpy as np
    x_data = torch.tensor(np.array([2.4, 3.6, 2.2, 1.8])).type(torch.float32)
    
    model = EEG_AE_MODEL(32, 1, 2, 3)

    import pdb;pdb.set_trace()
