#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   minimal version 1
            try mujoco released by deepmind
            run in my_work_env
@File   :   deepmind_mujoco_test1.py    
@author :   yinzikang

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
4/11/22 4:11 PM   yinzikang      1.0         None
"""
import mujoco as mj

def main():
    # 创建窗口
    max_width = 1000
    max_height = 1000
    mj.glfw.glfw.init()
    window = mj.glfw.glfw.create_window(max_width, max_height, "Demo", None, None)
    mj.glfw.glfw.make_context_current(window)
    mj.glfw.glfw.swap_interval(1)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    xml_path = "xml_analysis/UR5_gripper/UR5gripper.xml"
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    viewport = mj.MjrRect(0, 0, max_width, max_height)
    while not mj.glfw.glfw.window_should_close(window):
        simstart = data.time

        while (data.time - simstart < 1.0/60.0):
            mj.mj_step(model, data)

        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        mj.glfw.glfw.swap_buffers(window)
        mj.glfw.glfw.poll_events()

    mj.glfw.glfw.terminate()


if __name__ == "__main__":
    main()