from mujoco_py import load_model_from_path, MjSim, MjViewer


class my_ur5_env():
    def __init__(self):
        # super(lab_env, self).__init__(env)
        # 导入xml文档
        self.model = load_model_from_path("UR5_gripper/UR5gripper.xml")
        # 调用MjSim构建一个basic simulation
        self.sim = MjSim(model=self.model)
        self.env_viewer = MjViewer(self.sim)
        print(dir(self.env_viewer))
        print(self.env_viewer.__doc__)


    def get_state(self, *args):
        status = self.sim.get_state()
        return status

    def set_state(self, status):
        self.sim.set_state(status)

    def reset(self, *args):
        self.sim.reset()

    def step(self, *args):
        self.sim.step()

    def viewer(self, *args):
        self.env_viewer.render()
        self.env_viewer.add_marker()