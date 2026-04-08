import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Speedup Plot
# -----------------------------
df = pd.read_csv("scaling_results.csv")

plt.figure()
plt.plot(df["threads"], df["speedup"], marker='o', label="Actual")
plt.plot(df["threads"], df["threads"], linestyle='--', label="Ideal")
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.legend()
plt.grid()
plt.savefig("speedup.png", dpi=300)

# -----------------------------
# 2. Efficiency Plot
# -----------------------------
plt.figure()
plt.plot(df["threads"], df["efficiency"], marker='o')
plt.xlabel("Threads")
plt.ylabel("Efficiency")
plt.grid()
plt.savefig("efficiency.png", dpi=300)

# -----------------------------
# 3. Free Fall
# -----------------------------
df2 = pd.read_csv("test_freefall_traj.csv")

plt.figure()
plt.plot(df2["t"], df2["z_num"], label="Numerical")
plt.plot(df2["t"], df2["z_ana"], linestyle='--', label="Analytical")
plt.xlabel("Time")
plt.ylabel("Height")
plt.legend()
plt.grid()
plt.savefig("freefall.png", dpi=300)

# -----------------------------
# 4. Bounce
# -----------------------------
df3 = pd.read_csv("test_bounce_traj.csv")

plt.figure()
plt.plot(df3["t"], df3["z"])
plt.xlabel("Time")
plt.ylabel("Height")
plt.grid()
plt.savefig("bounce.png", dpi=300)