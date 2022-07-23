#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/30/22 5:38 PM   yinzikang      1.0         None
"""


def lengthOfLongestSubstring(s: str) -> int:
    table = [0] * 128
    left_index = 0
    max_length = 0
    cur_length = 0
    for right_index in range(len(s)):
        cur_length = right_index - left_index + 1
        table[ord(s[right_index])] += 1
        if table[ord(s[right_index])] > 1:
            cur_length = cur_length - 1
            max_length = cur_length if cur_length > max_length else max_length
            while s[left_index] is not s[right_index]:
                table[ord(s[left_index])] -= 1
                left_index += 1
            table[ord(s[left_index])] -= 1
            left_index += 1  # 定位到不同的第一个
            cur_length = 0
        else:
            max_length = cur_length if cur_length > max_length else max_length
        print(max_length, left_index, right_index)

    return max_length


print(lengthOfLongestSubstring('nfpdmpi'))