# inspect_shadow.py
import mujoco as mj

# Load the raw DeepMind hand model
model = mj.MjModel.from_xml_path("shadow_hand/right_hand.xml")

# Helpers
def get_name(obj_type, idx):
    # mjtObj enum lives in mujoco.mjtObj
    return mj.mj_id2name(model, getattr(mj.mjtObj, obj_type), idx)

print("\n--- JOINTS ---")
for i in range(model.njnt):
    print(f"  id={i:2d}  name='{get_name('mjOBJ_JOINT', i)}'")

print("\n--- ACTUATORS ---")
for i in range(model.nu):
    print(f"  id={i:2d}  name='{get_name('mjOBJ_ACTUATOR', i)}'")
