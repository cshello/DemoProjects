import torch 
import numpy as np 


from utils import net 
from utils import device 
net.to(device)
net.load_state_dict(torch.load("model.pth", map_location="cpu"))

sample = torch.randn(1, 3, 32, 32)

trace = torch.jit.trace(net, sample)

trace.save("model.jit")




