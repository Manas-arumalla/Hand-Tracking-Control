import cv2
import mediapipe as mp
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np

# -------- Hand‑tracking setup --------
mpHands = mp.solutions.hands
hands   = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils
finger_tips = [4, 8, 12, 16, 20]

# -------- MuJoCo setup --------
model = mj.MjModel.from_xml_path("cf2_scene.xml")
data  = mj.MjData(model)

# Initialize GLFW
if not glfw.init():
    raise RuntimeError("GLFW init failed")
window = glfw.create_window(800, 600, "Crazyflie Drone", None, None)
if not window:
    glfw.terminate(); raise RuntimeError("GLFW window failed")
glfw.make_context_current(window)

# -------- Camera setup --------
cam   = mj.MjvCamera(); opt = mj.MjvOption()
mj.mjv_defaultCamera(cam); mj.mjv_defaultOption(opt)
drone_body_id = model.body('cf2').id
cam.distance  = 0.5; cam.elevation = -20.0; cam.azimuth = 90.0
scene = mj.MjvScene(model, maxgeom=10000)
ctx   = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Map actuators
ctrl_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
THRUST_ID = ctrl_names.index('body_thrust')
ROLL_ID   = ctrl_names.index('y_moment')   # roll torque
PITCH_ID  = ctrl_names.index('x_moment')   # pitch torque

# Control parameters
base_thrust    = 0.26487   # hover thrust
vertical_delta = 0.3       # thrust change
moment_delta   = 2000       # torque magnitude

# Start webcam
cap = cv2.VideoCapture(0)
dt  = model.opt.timestep

while not glfw.window_should_close(window):
    # 1) Hand detection
    ret, frame = cap.read()
    if not ret: break
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res    = hands.process(imgRGB)
    count  = 0
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        h, w, _ = frame.shape
        for tip in finger_tips[1:]:
            if lm[tip].y * h < lm[tip-2].y * h:
                count += 1
        mpDraw.draw_landmarks(frame, res.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)

    # 2) Control signals (hover default)
    thrust = base_thrust
    roll   = 0.0
    pitch  = 0.0

    if count == 1:
        thrust = base_thrust + vertical_delta    # ascend
    elif count == 4:
        thrust = max(0.0, base_thrust - vertical_delta)  # descend
    elif count == 2:
        roll = -moment_delta   # tilt right
    elif count == 3:
        roll = moment_delta    # tilt left
    # hover (0 fingers): all zeros except hover thrust

    # 3) Apply controls
    data.ctrl[THRUST_ID] = thrust
    data.ctrl[ROLL_ID]   = roll
    data.ctrl[PITCH_ID]  = pitch

    # 4) Update camera to follow drone
    cam.lookat[:] = data.xpos[drone_body_id]

    # 5) Step and render
    mj.mj_step(model, data)
    vp = mj.MjrRect(0, 0, 800, 600)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(vp, scene, ctx)
    glfw.swap_buffers(window); glfw.poll_events()

    # 6) Display feedback
    cv2.putText(frame, f"Fingers: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Cleanup
cap.release(); cv2.destroyAllWindows(); glfw.terminate()