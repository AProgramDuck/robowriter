###
# Inverse Kinematics practice with MuJoCo
# SI100B Robotics Programming
# This code is modified based on the MuJoCo template code at https://github.com/pab47/pab47.github.io/tree/master.
# Date: Dec., 2025
###

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import scipy as sp

xml_path = '../universal_robots_ur5e/scene.xml' #xml file (assumes this is in the same folder as this file)
simend = 100 #simulation time (second)
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# Helper function
def random_cube_apex(
    cx, cy, cz,
    cube_size=0.1
):
    """
    Randomly returns the position of one apex of a cube in 3D space.

    Parameters
    ----------
    cube_size : float
        Edge length of the cube.
    center_range : tuple of 3 tuples
        Sampling range for cube center in x, y, z.

    Returns
    -------
    apex : np.ndarray, shape (3,)
        3D position of a randomly chosen cube apex.
    """
    # Random cube center
    center = np.array([cx, cy, cz])

    # Half edge length
    h = cube_size / 2.0

    # All 8 apex offsets
    offsets = np.array([
        [ h,  h,  h],
        [ h,  h, -h],
        [ h, -h,  h],
        [ h, -h, -h],
        [-h,  h,  h],
        [-h,  h, -h],
        [-h, -h,  h],
        [-h, -h, -h],
    ])

    # Randomly select one apex
    apex_offset = offsets[np.random.randint(0, 7)]
    apex = center + apex_offset

    return apex

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

# Get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1920, 1080, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = -130
cam.elevation = -5
cam.distance =  2
cam.lookat =np.array([ 0.0 , 0.0 , 0.5 ])

# Initialize the controller
init_controller(model,data)

# Set the controller
mj.set_mjcb_control(controller)

# Initialize home position
key_qpos = model.key("home").qpos
q_home = key_qpos.copy()

############################################################
##  TRY TO CHANGE THESE VALUES TO SEE DIFFERENT BEHAVIORS ##
##  REMEMBER TO KEEP WITHIN THE ROBOT'S WORKSPACE!        ##
############################################################
x_ref = 0.5
y_ref = 0.2
z_ref = 0.3
phi_ref = 3.14
theta_ref = 0
psi_ref = 0
############################################################

apex_ref = np.array([x_ref,y_ref,z_ref])
alter_flag = True

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        # Get current end-effector position and orientation
        position_Q = data.site_xpos[0]
        mat_Q = data.site_xmat[0]
        quat_Q = np.zeros(4)
        mj.mju_mat2Quat(quat_Q, mat_Q)
        r_Q = sp.spatial.transform.Rotation.from_quat([quat_Q[1],quat_Q[2],quat_Q[3],quat_Q[0]])
        euler_Q = r_Q.as_euler('xyz')
        
        # Compute Jacobian J
        # void mj_jac(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr,
        #     const mjtNum point[3], int body);
        jacp = np.zeros((3, 6))
        jacr = np.zeros((3, 6))
        mj.mj_jac(model, data, jacp, jacr, position_Q, 7)

        # Compute inverse Jacobian Jinv
        J = np.vstack((jacp, jacr))
        Jinv = np.linalg.pinv(J)

        # Compute dX
        if (int(data.time)%5==0) and (alter_flag==True):
            apex_ref = random_cube_apex(x_ref,y_ref,z_ref)
            alter_flag = False
        if (int(data.time)%5!=0):
            alter_flag = True
            
        X_ref = np.array([apex_ref[0],apex_ref[1],apex_ref[2],phi_ref,theta_ref,psi_ref])
        X = np.concatenate((position_Q, euler_Q))
        dX = X_ref - X

        # Compute dq = Jinv * dX
        dq = Jinv.dot(dX)

        q_home += dq
        data.qpos = q_home.copy()
        mj.mj_forward(model, data)
        data.time += 0.02

    if (data.time>=simend):
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()