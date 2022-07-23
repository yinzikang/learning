#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/30/22 9:53 PM   yinzikang      1.0         None
"""


def longestPalindrome(s: str) -> str:
    d_max_left = 0
    d_max_right = 0
    s_max_left = 0
    s_max_right = 0
    if len(s) == 1:
        return s
    for center in range(len(s) - 1):
        left = center
        right = center + 1
        while -1 < left and right < len(s) and s[left] is s[right]:
            left -= 1
            right += 1
            print(left, right)
        if right - left - 2 > d_max_right - d_max_left:
            d_max_left = left + 1
            d_max_right = right - 1

        left = center
        right = center
        while -1 < left and right < len(s) and s[left] is s[right]:
            left -= 1
            right += 1
            print(left, right)
        if right - left - 2 > s_max_right - s_max_left:
            s_max_left = left + 1
            s_max_right = right - 1
    return s[d_max_left:d_max_right+1] if d_max_right - d_max_left > s_max_right - s_max_left else s[ s_max_left:s_max_right+1]


print(longestPalindrome('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'))
