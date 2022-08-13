#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""利用mujoco-py测试equality的力学特性

发现有bug，传感器数值不对，因此换成了mujoco


Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
7/13/22 4:28 PM   yinzikang      1.0         None
"""
import numpy as np
import mujoco_py as mp

mjc_model = mp.load_model_from_path('connect_v4.xml')
mjc_model.opt.gravity[:] = np.array([0, 0, -10])
# mjc_model.eq_active[0] = 0
sim = mp.MjSim(model=mjc_model)
viewer = mp.MjViewer(sim)
viewer._paused = True

step = 0
duration = 2000

sim.forward()
while step < duration:
    # if step == 10:
    #     pos1 = sim.data.get_body_xpos("gripper").copy()
    #     pos2 = sim.data.get_body_xpos("box").copy()
    #     delta = np.hstack((pos1 - pos2, np.zeros(4)))
    #     sim.model.eq_data[0] = -delta
    #     sim.model.eq_active[0] = 1

    print(sim.data.sensordata[:])
    sim.step()
    viewer.render()
    step += 1
