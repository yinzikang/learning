#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""官方尚未支持高级的交互式ui，并推荐这个一个非官方版本

交互式ui的高级封装，用于显示以及加marker

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
8/6/22 9:55 AM   yinzikang      1.0         None
"""

import mujoco
import mujoco_viewer.mujoco_viewer as mujoco_viewer
import copy
import matplotlib.pyplot as plt
import numpy as np

model = mujoco.MjModel.from_xml_path(filename='xml_analysis/UR5_gripper/UR5gripper.xml')
data = mujoco.MjData(model)

init_state = copy.deepcopy(data)
pos = []

for j in range(20):
    mujoco.mj_resetData(model, data)
    data.joint('shoulder_pan_joint').qpos = 1.5
    mujoco.mj_forward(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)
    for i in range(500):
        data.ctrl[0] = 1
        mujoco.mj_step(model, data)
        pos.append(np.array(data.joint('shoulder_pan_joint').qpos))
        # print(data.sensordata[:])
        # print(data.site('gripperpalm').xmat)

        mass_matrix = np.ndarray(shape=(model.nv, model.nv), dtype=np.float64, order='C')
        mujoco.mj_fullM(model, mass_matrix, data.qM)
        # mass_matrix = np.reshape(mass_matrix, (model.nv, model.nv))
        mass_matrix = mass_matrix[:6, :6]
        # print(mass_matrix)

        jacr = np.ndarray(shape=(3, model.nv), dtype=np.float64, order='C')
        jacp = np.ndarray(shape=(3, model.nv), dtype=np.float64, order='C')
        mujoco.mj_jacSite(model, data, jacp, jacr, mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'gripperpalm'))
        print(jacp)
        print(jacr)

        viewer.render()

    viewer.close()

    plt.figure(1)
    plt.plot(np.array(pos))
    plt.show()
    pos = []
