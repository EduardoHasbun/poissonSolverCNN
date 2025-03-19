# Re-import necessary libraries since execution state was reset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
data = {
    "Case Name": [
        "1e-5_epoch_20", "1e-5_epoch_40", "1e-5_epoch_60", "1e-5_epoch_80", "1e-5_epoch_100"
    ],
    "Max Error (%)": [
        91.97, 93.66, 92.93, 92.60, 92.04
    ],
    "Avg Error (%)": [
        2.91, 2.83, 2.84, 2.85, 2.91
    ]
}


df = pd.DataFrame(data)

# Scale Max Error by dividing by 10 for better visualization
df["Max Error (%) Scaled"] = df["Max Error (%)"] / 10

# Plot
x = np.arange(len(df["Case Name"]))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, df["Max Error (%) Scaled"], width, label="Max Error (%) (Scaled / 10)", alpha=0.7)
bars2 = ax.bar(x + width/2, df["Avg Error (%)"], width, label="Avg Error (%)", alpha=0.7)

# Labels, title, and legend
ax.set_xlabel("Case Name")
ax.set_ylabel("Error (%)")
ax.set_title("Max and Avg Error for Different Architectures")
ax.set_xticks(x)
ax.set_xticklabels(df["Case Name"], rotation=45, ha="right")
ax.legend()

# Show plot
plt.tight_layout()
plt.show()
