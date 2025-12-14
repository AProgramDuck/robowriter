import math
import time

import mujoco
from mujoco import viewer
import numpy as np

# 场景：地面 + 带 free 关节的“杯子”（高圆柱 + RGB 轴）
MJCF = """
<mujoco>
  <option timestep="0.01"/>

  <asset>
    <!-- 
      定义一个中空圆柱体的网格。
      'vertex' 属性定义顶点坐标。
      'face' 属性定义连接顶点的三角面索引。
    -->
    <mesh name="cup_mesh"
          vertex="
            0.05 0 0      0.04 0 0      0.045 0.022 0   0.036 0.017 0
            0.035 -0.035 0  0.028 -0.028 0  0 -0.05 0     0 -0.04 0
            -0.035 -0.035 0 -0.028 -0.028 0 -0.05 0 0      -0.04 0 0
            -0.035 0.035 0  -0.028 0.028 0  0 0.05 0      0 0.04 0
            0.05 0 0.1     0.04 0 0.1     0.045 0.022 0.1  0.036 0.017 0.1
            0.035 -0.035 0.1 0.028 -0.028 0.1 0 -0.05 0.1    0 -0.04 0.1
            -0.035 -0.035 0.1 -0.028 -0.028 0.1 -0.05 0 0.1     -0.04 0 0.1
            -0.035 0.035 0.1 -0.028 0.028 0.1 0 0.05 0.1     0 0.04 0.1"
          face="
            0 2 1    2 3 1    2 4 3    4 5 3    4 6 5    6 7 5
            6 8 7    8 9 7    8 10 9   10 11 9   10 12 11 12 13 11
            12 14 13 14 15 13 14 0 15   0 1 15
            16 17 18   18 17 19   18 19 20   20 19 21   20 21 22   22 21 23
            22 23 24   24 23 25   24 25 26   26 25 27   26 27 28   28 27 29
            28 29 30   30 29 31   30 31 16   16 31 17
            0 16 2   16 18 2   2 18 4   18 20 4   4 20 6   20 22 6
            6 22 8   22 24 8   8 24 10  24 26 10  10 26 12  26 28 12
            12 28 14  28 30 14  14 30 16  30 0 16
            1 3 19   3 5 19   19 5 21   5 7 21   21 7 23   7 9 23
            23 9 25   9 11 25   25 11 27   11 13 27   27 13 29   13 15 29
            29 15 31   15 1 31   31 1 17   1 17 19"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- 杯子主体：由杯底、杯壁、把手组合而成 -->
    <body name="cup" pos="0 0 0.1">
      <!-- free 关节：位置 + 姿态都由 qpos 控制 -->
      <joint name="cup_free" type="free"/>

      <!-- 杯身：使用上面定义的网格 -->
      <geom type="mesh" mesh="cup_mesh" rgba="0.8 0.8 1 1"/>

      <!-- 把手：一个环面 -->
      <!-- 把手：用 3 根胶囊近似成 C 形 -->
      <geom name="handle_vert"   type="capsule" size="0.006 0.025" pos="0.06 0 0.05" rgba="0.8 0.8 1 1"/>
      <geom name="handle_top"    type="capsule" size="0.006 0.015" pos="0.06 0 0.08" euler="90 0 0" rgba="0.8 0.8 1 1"/>
      <geom name="handle_bottom" type="capsule" size="0.006 0.015" pos="0.06 0 0.02" euler="90 0 0" rgba="0.8 0.8 1 1"/>
      <!-- 显示局部坐标轴：X 红、Y 绿、Z 蓝 -->
      <geom type="cylinder" size="0.005 0.3" pos="0.3 0 0" euler="0 90 0" rgba="1 0 0 1"/>
      <geom type="cylinder" size="0.005 0.3" pos="0 0.3 0" euler="90 0 0" rgba="0 1 0 1"/>
      <geom type="cylinder" size="0.005 0.3" pos="0 0 0.3" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def axis_angle_to_quat(axis: str, angle_deg: float) -> np.ndarray:
    """绕单个坐标轴旋转 angle_deg（度），返回四元数 [qw, qx, qy, qz]."""
    half = math.radians(angle_deg) / 2.0
    c = math.cos(half)
    s = math.sin(half)
    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=float)
    elif axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=float)
    elif axis == "z":
        return np.array([c, 0.0, 0.0, s], dtype=float)
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 q = q1 ⊗ q2，均为 [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=float)

def slerp(q_from: np.ndarray, q_to: np.ndarray, t: float) -> np.ndarray:
    """四元数球面插值，t∈[0,1]."""
    q_from = q_from / np.linalg.norm(q_from)
    q_to = q_to / np.linalg.norm(q_to)
    dot = float(np.dot(q_from, q_to))
    if dot < 0.0:  # 走最短弧
        q_to = -q_to
        dot = -dot
    if dot > 0.9995:  # 夹角很小时用线性插值
        q = q_from + t * (q_to - q_from)
        return q / np.linalg.norm(q)
    theta0 = math.acos(dot)
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return s0 * q_from + s1 * q_to

def main() -> None:
  # 1. 建模 + 初始姿态
  model = mujoco.MjModel.from_xml_string(MJCF)
  data = mujoco.MjData(model)

  data.qpos[0:3] = [0.0, 0.0, 0.5]           # 位置
  q0 = np.array([1.0, 0.0, 0.0, 0.0])        # 初始单位四元数
  data.qpos[3:7] = q0
  mujoco.mj_forward(model, data)

  # 目标角度
  target_y = 45.0
  target_x = 30.0
  target_z = 60.0

  print("初始姿态 q0 =", q0)

  # 2. 阶段 0：绕“自身 y 轴”转 45°
  q_step_y = axis_angle_to_quat("y", target_y)
  q1 = quat_mul(q0, q_step_y)   # 右乘 -> 自身轴
  q1 /= np.linalg.norm(q1)
  print("阶段0（自身 y 轴 45°）后的 q1 =", q1)
  # 3. 阶段 1：在当前姿态基础上，绕“自身 x 轴”转 30°
  q_step_x = axis_angle_to_quat("x", target_x)
  q2 = quat_mul(q1, q_step_x)
  q2 /= np.linalg.norm(q2)
  print("阶段1（自身 x 轴 30°）后的 q2 =", q2)
  # 4. 阶段 2：在当前姿态基础上，绕“自身 z 轴”转 60°
  q_step_z = axis_angle_to_quat("z", target_z)
  q3 = quat_mul(q2, q_step_z)
  q3 /= np.linalg.norm(q3)
  print("阶段2（自身 z 轴 60°）后的 q3 =", q3)

  # 预先把四个阶段的姿态排好：0=初始,1,2,3
  quats = [q0, q1, q2, q3]

  transition_time = 1.5  # 每段旋转用时（秒）
  stage = 1              # 当前目标阶段（从 q0→q1 开始转）
  stage_start = time.time()

  with viewer.launch_passive(model, data) as v:
    while v.is_running():
      now = time.time()
      if stage <= 3:
        q_from = quats[stage - 1]
        q_to = quats[stage]
        alpha = min((now - stage_start) / transition_time, 1.0)
        current_q = slerp(q_from, q_to, alpha)
        if alpha >= 1.0 and stage < 3:
          stage += 1
          stage_start = now
      else:
        current_q = quats[-1]

      data.qpos[3:7] = current_q
      mujoco.mj_forward(model, data)
      v.sync()

if __name__ == "__main__":
    main()