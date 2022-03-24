import torch as th

print(th.cuda.is_available())
print(th.cuda.current_device())
print(dir(th))

tensor_1 = th.tensor([1, 1, 3, 4])
print(tensor_1)
print(tensor_1.shape)
# print(th.Size(tensor_1))
print(tensor_1.__class__)

tensor_2 = tensor_1.reshape(2, 2)
print(tensor_2)
