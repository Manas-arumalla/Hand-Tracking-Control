import cv2
import mediapipe as mp
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np

# ——————— Hand‑tracking setup ———————
mpHands = mp.solutions.hands
hands   = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw  = mp.solutions.drawing_utils

finger_tips_ids = [4, 8, 12, 16, 20]
finger_names    = ['Thumb','Index','Middle','Ring','Pinky']

# ——————— MuJoCo setup ———————
model = mj.MjModel.from_xml_path("shadow_hand_scene.xml")
data  = mj.MjData(model)

# GLFW window for rendering
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")
window = glfw.create_window(800, 600, "Shadow Hand", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Failed to create GLFW window")
glfw.make_context_current(window)

# Use default camera and option defined in scene_right.xml
cam   = mj.MjvCamera()
opt   = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
ctx   = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Initialize camera transform
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
cam.lookat    = np.array([0.0, 0.0, 0.0])
cam.distance  = 0.4     # default distance
cam.elevation = -20.0
cam.azimuth   = 120.0

# ——————— Manual view control ———————
# Arrow keys: rotate azimuth/elevation; +/-: zoom
def key_callback(window, key, scancode, action, mods):
    if action != glfw.PRESS and action != glfw.REPEAT:
        return
    global cam
    if key == glfw.KEY_LEFT:
        cam.azimuth -= 5.0
    elif key == glfw.KEY_RIGHT:
        cam.azimuth += 5.0
    elif key == glfw.KEY_UP:
        cam.elevation += 5.0
    elif key == glfw.KEY_DOWN:
        cam.elevation -= 5.0
    elif key == glfw.KEY_EQUAL:  # '+' key
        cam.distance *= 0.9
    elif key == glfw.KEY_MINUS:
        cam.distance *= 1.1

glfw.set_key_callback(window, key_callback)

# ——————— Mapping fingers → actuator IDs ———————
finger_actuators = {
  'Thumb': [2, 3, 4, 5, 6],    # rh_A_THJ5, THJ4, THJ3, THJ2, THJ1
  'Index': [7, 8],             # rh_A_FFJ4, FFJ3
  'Middle':[10,11],            # rh_A_MFJ4, MFJ3
  'Ring':  [13,14],            # rh_A_RFJ4, RFJ3
  'Pinky': [16,17,18],         # rh_A_LFJ5, LFJ4, LFJ3
}

# Angle mapping: 0 rad=open, 1.2 rad=closed
open_angle   = 0.0
closed_angle = 1.2

# Initialize camera capture
cap = cv2.VideoCapture(0)

dt  = model.opt.timestep

while not glfw.window_should_close(window):
    # 1) Read frame + detect fingers
    ret, frame = cap.read()
    if not ret:
        break
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res    = hands.process(imgRGB)
    lowered = []

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        pts = [(i, int(p.x*frame.shape[1]), int(p.y*frame.shape[0]))
               for i,p in enumerate(lm)]

        # Thumb: x-axis test
        if pts[4][1] < pts[3][1]:
            lowered.append('Thumb')
        # Other four: y-axis test
        for i in range(1,5):
            tip = finger_tips_ids[i]
            if pts[tip][2] > pts[tip-2][2]:
                lowered.append(finger_names[i])

        mpDraw.draw_landmarks(frame,
                              res.multi_hand_landmarks[0],
                              mpHands.HAND_CONNECTIONS)

    # 2) Apply target angles
    for finger in finger_names:
        target = closed_angle if finger in lowered else open_angle
        for aid in finger_actuators[finger]:
            data.ctrl[aid] = target

    # 3) Step + render MuJoCo
    mj.mj_step(model, data)
    viewport = mj.MjrRect(0, 0, 800, 600)
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, ctx)
    glfw.swap_buffers(window)
    glfw.poll_events()

    # 4) Show webcam overlay
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
glfw.terminate()