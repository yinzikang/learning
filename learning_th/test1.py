#  gpu检测程序
import torch

print(torch.cuda.is_available())
print(torch.cuda.current_device())
device = torch.device("cuda")
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())

a = torch.randn(2, 2)
b = torch.randn(2, 2).to(device)
c = torch.randn(2,2,requires_grad=True)
d = torch.randn(2,2,requires_grad=True)
e = c + d
print(a)
print(b)
print(c)
print(e)