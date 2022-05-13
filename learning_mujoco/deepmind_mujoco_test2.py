#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   minimal version 2
            try mujoco released by deepmind
            run in my_work_env
@File   :   deepmind_mujoco_test1.py    
@author :   yinzikang

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
4/11/22 4:11 PM   yinzikang      1.0         None
"""
import mujoco
import glfw

def init_window(max_width, max_height):
    glfw.init()
    window = glfw.create_window(width=max_width, height=max_height,
                                       title='Demo', monitor=None,
                                       share=None)
    glfw.make_context_current(window)
    return window

window = init_window(1200, 900)

model = mujoco.MjModel.from_xml_path('xml_analysis/UR5_gripper/UR5gripper.xml')
data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

width, height = glfw.get_framebuffer_size(window)
viewport = mujoco.MjrRect(0, 0, width, height)

scene = mujoco.MjvScene(model, 1000)
camera = mujoco.MjvCamera()
mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

while(not glfw.window_should_close(window)):
    mujoco.mj_step(model, data)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
