#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
6/23/22 12:01 PM   yinzikang      1.0         None
"""
import PyKDL as kdl
import numpy as np


def getForwardKinematics(rbt, joint_pos):
    fk = kdl.ChainFkSolverPos_recursive(rbt)
    cart_pos_ori = kdl.Frame()
    fk.JntToCart(joint_pos, cart_pos_ori)
    return cart_pos_ori


def getInverseKinematics(rbt, joint_init_pos, cart_pos_ori):
    ik = kdl.ChainIkSolverPos_LMA(rbt, maxiter=1500)
    joint_pos = kdl.JntArray(2)
    ik.CartToJnt(joint_init_pos, cart_pos_ori, joint_pos)
    return joint_pos


# 以下机器人为一个二连杆机器人放置在桌子上
# 桌高+连杆0高3m，连杆1,2长1m，末端执行器长1m
jnts = []
frms = []
links = []
jnts.append(kdl.Joint("joint0", kdl.Joint.Fixed))  # 连杆0与桌子链接的地方，必然为fixed
jnts.append(kdl.Joint("joint1", kdl.Joint.RotZ))  # 关节1,第一个电机，一般为绕z旋转
jnts.append(kdl.Joint("joint2", kdl.Joint.RotY))  # 关节n，最后一个电机
jnts.append(kdl.Joint("joint_end", kdl.Joint.Fixed))  # 连杆n与执行器相接的地方，必然为fixed
frms.append(kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 3)))  # 连杆0的长度（+桌子高度），即连杆1近端的位置
frms.append(kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 1)))  # 连杆1的长度
frms.append(kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 1)))  # 连杆n的长度
frms.append(kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 1)))  # 执行器长度
links.append(kdl.Segment("link_0", jnts[0], frms[0]))
links.append(kdl.Segment("link_1", jnts[1], frms[1]))
links.append(kdl.Segment("link_n", jnts[2], frms[2]))
links.append(kdl.Segment("end_effector", jnts[3], frms[3]))

rbt = kdl.Chain()
for link in links:
    rbt.addSegment(link)

# 对于Fixed关节，无法调节，长度为所有活动关节
qpos_init = kdl.JntArray(2)
qpos_init[1] = -np.pi / 2
xpos = getForwardKinematics(rbt, qpos_init)
print(xpos.p)

qpos =  getInverseKinematics(rbt, qpos_init, xpos)
print(qpos)


