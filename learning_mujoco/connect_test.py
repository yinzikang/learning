#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
7/13/22 11:39 AM   yinzikang      1.0         None
"""
import numpy as np
import mujoco_py as mp

mjc_model = mp.load_model_from_path('connect.xml')
# mjc_model.qpos0[0] = 30
sim = mp.MjSim(model=mjc_model)
viewer = mp.MjViewer(sim)
# sim.data.qpos[0] = 0.5
# sim.forward()

while True:
    sim.data.qpos[0] += 0.001
    sim.step()
    viewer.render()