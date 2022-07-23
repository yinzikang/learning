import torch as th

# print(th.cuda.is_available())
# print(th.cuda.current_device())
# print(dir(th))
#
# tensor_1 = th.tensor([1, 1, 3, 4])
# print(tensor_1)
# print(tensor_1.shape)
# # print(th.Size(tensor_1))
# print(tensor_1.__class__)
#
# tensor_2 = tensor_1.reshape(2, 2)
# print(tensor_2)

tensor_3 = th.randn([2, 5, 3])
tensor_4 = [2, 3]
tensor_5 = th.zeros_like(tensor_3)
tensor_6 = th.zeros_like(tensor_3)
for i in range(tensor_3.shape[0]):
    tensor_5[i, :tensor_4[i], :] = tensor_3[i, :tensor_4[i], :]
tensor_6[:, :tensor_4, :] = tensor_3[:, :tensor_4, :]
print(tensor_3)
print(tensor_5)
print(tensor_5 - tensor_6)
