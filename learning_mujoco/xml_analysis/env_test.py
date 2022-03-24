import ur5_env
import numpy as np

my_env = ur5_env.my_ur5_env()
status = my_env.get_state()
duration = 4

id = my_env.sim.model.site_name2id('ee')
# step = 0
pos = []
while status.time < 4:
    # status = my_env.get_state()
    # status.qpos[0] = status.time
    # my_env.set_state(status)

    data = my_env.sim.data

    my_env.step()
    my_env.viewer()
    pos.append(data.site_xpos[id])
    # my_env.env_viewer.add_marker(pos=np.array(pos), label='ee')
    my_env.env_viewer.add_marker(pos=np.array(data.site_xpos[id]), label='ee')
    print(my_env.sim.data.time)

