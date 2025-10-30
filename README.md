# ⚛️ Laboratory Work: Computational Methods with Quantum ESPRESSO
**Course:** Computational Methods for Atomistic Simulations <br>
**Authors:** Khasan Akhmadiev, Alexander Kvashnin <br>
**Purpose:** This repository contains laboratory assignments for the course, focusing on calculating phonon / electronic properties of materials using the **Quantum ESPRESSO** (QE) package.

---

## 🚀 Quick Start (Setup and Execution)

This project is primarily designed for running calculations on a **remote server** (e.g., `HeiMao server`).

### 1. Repository Cloning
1.  Create a working directory on the server.
2.  Clone the repository:
    ```bash
    git clone [https://github.com/hasaki77/QE_Laboratory_Work.git](https://github.com/hasaki77/QE_Laboratory_Work.git)
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
| **SFTP Client** | WinSCP / FileZilla | | Managing files between local machine and server. |

Good luck with your simulations! 🍀

Hello everyone! 👋
✅ 1) Check if you have these or similar programs:

▪️ SSH Client (e.g. Putty) will be used to connect to the server:
  https://putty.org/index.html

▪️ Code editor with jupyter-notebook (e.g. VScode) will be used to run code:
  https://code.visualstudio.com/download

▪️ A client (e.g. WinSCP for Windows) to work with files between local computer and remote server

✅ 2) Access to work on remote server:
- Connect to Skoltech Wi-Fi
- Go to terminal (e.g. Putty)
- HostName: heimao@10.16.68.25
- Password: hei_mao25

✅ 3) Before each job execution, please:
▪️  Make sure that local environment is activated
conda activate lab_env
▪️ Make sure that mpirun is working:
source ~/intel/oneapi/setvars.sh > /dev/null
▪️ The commands on the shell is for HeiMao server. If you use your own server, please, make correct commands, accordingly.

✅ 4) Connect your VScode to the server:
- If you do it for the first time, please, google how to do it
- If you have experience, then use:
Ctrl+Shift+P -> Remote-SSH: Connect to Host -> Add new SSH Host -> ssh heimao@10.16.68.25

✅ 5) Start with Lab Work:
- Create a folder in working directory with your Name (for HeiMao: create folder in Desktop directory)
- Go to the teminal and copy repository: 
git clone https://github.com/hasaki77/QE_Laboratory_Work.git
- Go to VScode and open jupyter-notebook in the cloned repository.
- Connect to python local environment: Select Kernel -> lab_env
- Try to run the cell with python libraries. If it works, job successfully done! Congrats! 🥳

✅ 6) Choose the two structures:
- The first structure must be taken from github repo. It will be 2D material. This material will be your Lab Work.
- The second structure has to be your own choice. This material will be your Project, that you will defend.
📌 Please, mention chosen structure in telegram chat in order not to be repeated with other students.

If you have done all above steps, you can start working with labs.

Content: <br>
🟣 [**Lab Assignment 1**: Structure Optimization](1_Structure_Optimization.ipynb) <br>
🎯**Objective:** Optimize cut-off energy and k-points grid parameters and implement them for structure relaxation.

**General Workflow:** <br>
- Import and Initialization
- Initialize functions and download essential files
- Implement SCF calculations for cut-off energy optimization
- Implement SCF calculations for k-points grid optimization
- Implement structure relaxation with optimized parameters

🟣 [Lab Assignment 2: Phonon & Electronic Properties](1_Structure_Optimization.ipynb) <br>
🎯**Objective:** Calculate phonon/electronic density of states and band structure.

**General Workflow:** <br>
- Import and Initialization
- Initialize functions and download essential files
- Implement SCF calculation and obtain phonon DOS
- Implement SCF calculation and obtain phonon BAND structure
- Implement SCF & NSCF calculations and obtain electronic DOS
- Implement SCF calculation and obtain electronic BAND structure

🟣 [Lab Assignment 3: MLIP Training](1_Structure_Optimization.ipynb) <br>
🎯**Objective:** Implement molecular dynamics to obtain structures that will be used to train Machine Learning Interatomic Potential (MLIP).

**General Workflow:** <br>
- Import and Initialization
- Initialize functions and download essential files
- Implement MD calculation
- Train MLIP model

Good luck! 🍀
