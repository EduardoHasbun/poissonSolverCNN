import matplotlib.pyplot as plt

def extract_total_losses(file_path, weight, label):
    with open(file_path, "r") as file:
        lines = file.readlines()

    total_losses = []
    reading_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "Laplacian Losses:" in line:
            reading_section = "total"
            continue
        elif any(header in line for header in ["Epoch Losses:", "Batch Losses:", "Laplaci Losses:", "Dirichlet Losses:"]):
            reading_section = None
            continue

        if reading_section == "total":
            try:
                values = [float(x) / weight for x in line.split(",") if x.strip()]
                total_losses.extend(values)
            except ValueError:
                print(f"Skipping malformed line: {line}")
    
    return total_losses

# Ruta de los archivos y etiquetas personalizadas
file1 = "C:/Codigos/poissonSolverCNN/2D/training/trained_models/Unet4_ks3_rf200_opti_1e-3.pth_losses.txt"
file2 = "C:/Codigos/poissonSolverCNN/2D/training/trained_models/Unet4_ks3_rf200_opti_1e-5.pth_losses.txt"
label1 = "Adam = 1e-3"
label2 = "Adam = 1e-5"

# Pesos para normalizar (ajusta según corresponda)
total_weight = 8.0e+5

# Extrae pérdidas totales normalizadas
total_losses1 = extract_total_losses(file1, total_weight, label1)
total_losses2 = extract_total_losses(file2, total_weight, label2)

# Recorta a un número máximo de pasos si es necesario
max_steps = min(len(total_losses1), len(total_losses2), 5000)
total_losses1 = total_losses1[:max_steps]
total_losses2 = total_losses2[:max_steps]

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(total_losses1, label=label1)
plt.plot(total_losses2, label=label2)
plt.xlabel("Step")
plt.ylabel("Total Loss (normalized)")
plt.title("Total Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
