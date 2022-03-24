from dm_control import mjcf
import numpy as np


class Leg(object):
    """ A 2-DoF leg with position actuators."""

    def __init__(self, length, rgba):
        self.model = mjcf.RootElement()
        # Defaults:
        self.model.default.joint.damping = 2
        self.model.default.joint.type = 'hinge'
        self.model.default.geom.type = 'capsule'
        self.model.default.geom.rgba = rgba  # Continued below...
        # Thigh:
        self.thigh = self.model.worldbody.add('body')
        self.hip = self.thigh.add('joint', axis=[0, 0, 1])
        self.thigh.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[length / 4])
        # Shin:
        self.shin = self.thigh.add('body', pos=[length, 0, 0])
        self.knee = self.shin.add('joint', axis=[0, 1, 0])
        self.shin.add('geom', fromto=[0, 0, 0, 0, 0, -length], size=[length / 5])
        # Position actuators:
        self.model.actuator.add('position', joint=self.hip, kp=10)
        self.model.actuator.add('position', joint=self.knee, kp=10)

BODY_RADIUS = 0.1
BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)

def make_creature(num_legs):
    """Constructs a creature with ‘num_legs‘ legs."""
    rgba = np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1])
    model = mjcf.RootElement()
    model.compiler.angle = 'radian'  # Use radians.
    # Make the torso geom.
    torso = model.worldbody.add('geom', name='torso', type='ellipsoid', size=BODY_SIZE, rgba=rgba)
    # Attach legs to equidistant sites on the circumference.
    for i in range(num_legs):
        theta = 2 * i * np.pi / num_legs
        hip_pos = BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
        hip_site = model.worldbody.add('site', pos=hip_pos, euler=[0, 0, theta])
        leg = Leg(length=BODY_RADIUS, rgba=rgba)
        hip_site.attach(leg.model)
    return model

arena = mjcf.RootElement()
checker = arena.asset.add('texture', type='2d', builtin='checker', width=300, height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
grid = arena.asset.add('material', name='grid', texture=checker, texrepeat=[5, 5], reflectance=.2)
arena.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)
for x in [-2, 2]:
    arena.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2])
# Instantiate 6 creatures with 3 to 8 legs.
creatures = [make_creature(num_legs=num_legs) for num_legs in (3, 4, 5, 6, 7, 8)]
# Place them on a grid in the arena.
height = .15
grid = 5 * BODY_RADIUS
xpos, ypos, zpos = np.meshgrid([-grid, 0, grid], [0, grid], [height])
for i, model in enumerate(creatures):
    # Place spawn sites on a grid.
    spawn_pos = (xpos.flat[i], ypos.flat[i], zpos.flat[i])
    spawn_site = arena.worldbody.add('site', pos=spawn_pos, group=3)
    # Attach to the arena at the spawn sites, with a free joint.
    spawn_site.attach(model).add('freejoint')
# Instantiate the physics and render.
physics = mjcf.Physics.from_mjcf_model(arena)
pixels = physics.render()
print(pixels)