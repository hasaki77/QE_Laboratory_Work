# ⚛️ Laboratory Work: Computational Methods with Quantum ESPRESSO
**Course:** Computational Methods for Atomistic Simulations <br>
**Authors:** Khasan Akhmadiev, Alexander Kvashnin <br>
**Purpose:** This repository contains laboratory assignments for the course, focusing on calculating phonon / electronic properties of materials using the **Quantum ESPRESSO** (QE) package.

---

## ⚙️ Setup and Execution

This project is primarily designed for running calculations on a **remote server**.

### 1. Repository Cloning
1.  Create a working directory on the server.
2.  Clone the repository:
    ```bash
    git clone https://github.com/hasaki77/QE_Laboratory_Work.git
    cd QE_Laboratory_Work
    ```

### 2. Environment Setup (Conda)
An isolated environment is required to run the Jupyter Notebooks.
1.  Install the necessary packages using the dependency file (`environment.yml`):
    ```bash
    conda env create -f environment.yml
    conda activate lab_env
    ```
2.  **Activating mpirun**
    Before executing any calculation, ensure the QE environment variables are correctly sourced. Check, that MPI is working:
    ```bash
    mpirun --version
    ```
    
### 3. Running Lab Assignments
1.  Connect to the server using **VSCode Remote-SSH** (recommended) or open a Jupyter Notebook server session.
2.  Open the lab assignment file (e.g., `1_Structure_Optimization.ipynb`).
3.  Inside the Notebook, **select the kernel** `lab_env`.
4.  Run all cells to confirm functionality.
---

## 📌 Lab Assignments and Objectives

Each lab assignment is presented as a separate Jupyter Notebook, which includes a theoretical introduction, calculation code, and specific tasks.

| No. | Assignment Title | Primary Objective | File Link |
| :---: | :--- | :--- | :--- |
| **1** | **Structure Optimization** | Optimize cut-off energy and k-points parameters, followed by structure relaxation. | [`1_Structure_Optimization.ipynb`](1_Structure_Optimization.ipynb) |
| **2** | **Phonon & Electronic Properties** | Calculate phonon/electronic Density of States (DOS) and band structures. | [`2_Phonon_Electronic_Properties.ipynb`](2_Phonon_Electronic_Properties.ipynb) |
| **3** | **MLIP Training** | Implement Molecular Dynamics to generate structures used for training a Machine Learning Interatomic Potential (MLIP). | [`3_MLIP_Training.ipynb`](3_MLIP_Training.ipynb) |

### 🔬 Project Assignment
In addition to the labs, students must select a **second structure** (of their own choice) to implement the **Final Project**.

---

## 💻 Recommended Software for Connection

| Category | Software | Link | Purpose |
| :--- | :--- | :--- | :--- |
| **SSH Client** | PuTTY / Built-in Terminal | [putty.org](https://putty.org/index.html) | Remote connection to the server. |
| **Code Editor** | **VSCode** with Remote-SSH | [code.visualstudio.com](https://code.visualstudio.com/download) | Code editing and running Jupyter Notebooks. |
| **SFTP Client** | WinSCP / FileZilla | [winscp.net](https://winscp.net/eng/index.php) | Managing files between local machine and server. |

Good luck with your calculations! 🍀
