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

model = mujoco.MjModel.from_xml_path(filename='connect_v4.xml')
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)

for i in range(1000):
    mujoco.mj_step(model, data)
    print(data.sensordata[:])

    viewer.render()
viewer.close()
