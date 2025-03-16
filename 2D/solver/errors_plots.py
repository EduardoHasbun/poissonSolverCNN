# Re-import necessary libraries since execution state was reset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
data = {
    "Case Name": [
        "Unet4_ks3_rf50", "Unet3_ks3_rf50", "Unet3_ks3_rf100", "Unet4_ks3_rf100",
        "Unet4_ks3_rf200", "Unet3_ks3_rf200", "Unet3_ks3_rf300", "Unet4_ks3_rf300", 
        "Unet3_ks3_rf400", "Unet4_ks3_rf400", "MSNet3_ks3_rf50", "MSNet4_ks3_rf50",
        "MSNet3_ks3_rf100", "MSNet3_ks3_rf200", "MSNet4_ks3_rf200"
    ],
    "Max Error (%)": [93.81, 88.87, 270.79, 226.82, 98.83, 169.87, 89.81, 101.50, 89.03, 88.83, 127.00, 72.85, 86.34, 96.57, 72.43],
    "Avg Error (%)": [2.98, 3.83, 14.36, 7.99, 3.35, 15.12, 3.58, 3.91, 3.66, 3.52, 3.03, 3.02, 3.00, 2.85, 3.44]
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
