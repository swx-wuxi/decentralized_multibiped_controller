import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eval_logs/log.csv")

# 只看第一个episode
ep = df[df["episode"] == 3]
target_yaw = ep["target_yaw"].values[0]
# ===== yaw tracking =====
plt.figure()
plt.plot(ep["step"], ep["target_yaw"], label="target")
plt.plot(ep["step"], ep["current_yaw"], label="current")
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("yaw (deg)")
plt.title("Orientation Tracking")
plt.show()

# ===== lateral drift =====
plt.figure()
plt.plot(ep["step"], ep["delta_y"])
plt.xlabel("time (s)")
plt.ylabel("Δy (m)")
plt.title("Lateral Drift")
plt.show()

# ===== trajectory =====
plt.figure()
plt.plot(ep["pos_x"], ep["pos_y"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory")
plt.axis("equal")
plt.show()