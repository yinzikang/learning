#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""learn to use linked list in python

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/30/22 11:08 AM   yinzikang      1.0         None
"""


class ListNode:
    def __init__(self, data=None):
        self.node_data = data
        self.next_node = None
        self.pre_node = None


class LinkedList:
    def __init__(self):
        self.head_node = None

    def list_print(self):
        current_node = self.head_node
        while current_node:
            print(current_node.node_data)
            current_node = current_node.next_node

    def delete_node(self, val):
        """
        在链表开头加入一个虚拟node，链表扩充，然后不断往后查删，最后把开头加的删了
        :param val:
        :return:
        """
        virtual_node = ListNode()
        virtual_node.next_node = self.head_node
        self.head_node = virtual_node
        while virtual_node.next_node:
            if virtual_node.next_node.node_data == val:
                virtual_node.next_node = virtual_node.next_node.next_node
            else:
                virtual_node = virtual_node.next_node
        self.head_node = self.head_node.next_node

    def inverse(self):
        left_node = self.head_node
        right_node = left_node.next_node
        left_node.next_node = None
        while right_node.next_node:
            temp_node = right_node.next_node
            right_node.next_node = left_node
            left_node = right_node
            right_node = temp_node
        right_node.next_node = left_node
        self.head_node = right_node


node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)

node1.next_node = node2
node2.next_node = node3
node3.next_node = node4

linked_list_1 = LinkedList()
linked_list_1.head_node = node1
# linked_list_1.delete_node(1)
# linked_list_1.list_print()
linked_list_1.inverse()
linked_list_1.list_print()
