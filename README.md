# 🧠 Convolutional Neural Network Solver for the Poisson Equation

This repository provides a **CNN-based solver** for the Poisson equation.  
The model learns the **inverse operator** of the equation, enabling fast solutions for new cases without retraining — as long as they share the same domain and boundary conditions.

---

## ✨ Features
- Solves the Poisson equation in **2D** and **3D**.
- Learns the inverse operator through supervised training.
- Once trained, the model can approximate solutions for new inputs **without solving from scratch**.
- Modular design with separate folders for **dataset generation**, **training**, and **solver usage**.

---

## ⚠️ Limitations
- During training, the user must define the **domain** and **boundary conditions**.
- The trained model is only valid **within the same domain and boundary conditions**.  
  For different setups, **retraining is required**.

---

## 📂 Project Structure

project-root/

│── 2D/ or 3D/            # Separate implementations for 2D and 3D cases

│   ├── dataset/          # Scripts to generate training data

│   ├── training/         # Scripts for model training

│   └── solver/           # Scripts to run the trained model

│
└── README.md             # Project documentation





---


## 🚀 Usage

### 1. Dataset Generation
Go to the `dataset` folder (inside `2D/` or `3D/`) and run: python dataset.py -c dataset.yml



### 2. Training
Go to the `training` folder and run: python train.py -c train.yml


> Modify `train.yml` to adjust training parameters.

### 3. Solving with the Trained Model
Go to the `solver` folder:
1. Edit `solver.yml`:
   - Set the path to the trained model
   - Define the case to solve  
2. Run: python solver.py


---


### 3. Citation
@misc{cnn-poisson-solver,

  title={Convolutional Neural Network Solver for the Poisson Equation},
  
  author={Eduardo Hasbun},
  
  year={2025},
  
  howpublished={GitHub repository},
  
  url={[https://github.com/your-repo-link](https://github.com/EduardoHasbun/poissonSolverCNN)}
  
}

