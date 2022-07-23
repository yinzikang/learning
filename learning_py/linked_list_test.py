#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""learn to use node in python

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/30/22 10:16 AM   yinzikang      1.0         None
"""


class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)

node1.next = node2
node2.next = node3
node3.next = node4

current_node = node1
while current_node:
    print(current_node.val)
    current_node = current_node.next


