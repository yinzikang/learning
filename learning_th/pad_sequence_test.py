#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/20/22 2:24 PM   yinzikang      1.0         None
"""
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

origin_array = [torch.Tensor([1, 2, 3, 4, 5, 6]),
                torch.Tensor([7, 8]),
                torch.Tensor([9])]
origin_length = [origin_array[i].shape[0] for i in range(0, len(origin_array))]
# input list of tensors
# return a tensor with batchsize*time_length*feature_dim
padded_array = pad_sequence(sequences=origin_array, batch_first=True, padding_value=0.0)
# input a tensor
# return a PackedSequence
packed_array1 = pack_padded_sequence(input=padded_array, lengths=origin_length, batch_first=True, enforce_sorted=False)
packed_array2 = pack_sequence(sequences=origin_array, enforce_sorted=False)
packed_array3 = pack_padded_sequence(input=padded_array[:, 3:], lengths=[0, 0, 3], batch_first=True, enforce_sorted=False)
# input a PackedSequence
# output a tuple: a tensor same as origin_array, and a tensor like origin_length
last_array1, last_length1 = pad_packed_sequence(sequence=packed_array1, batch_first=True)
last_array2, last_length2 = pad_packed_sequence(sequence=packed_array2, batch_first=True)

while True:
    pass
