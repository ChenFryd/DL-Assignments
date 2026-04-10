# Assignment 1 — Neural Network from Scratch

Build a fully-connected neural network from scratch using NumPy, without any deep learning frameworks. The goal is to gain a deep understanding of forward and backward propagation.

**Sections:**
1. **Forward Propagation** — parameter initialization, linear forward, ReLU/Softmax activations, batch normalization, and cost computation
2. **Backward Propagation** — gradients for linear, ReLU, and Softmax layers, and parameter updates
3. **Full Training Loop** — L-layer model training with mini-batches and a predict function
4. **Experiments** — hyperparameter tuning and analysis
5. **Conclusions**

---

# Extensions
Please install the `Python` and `Jupyter` extensions

# How to Install
1. Have Python > 3.9 and Git
```bash
sudo apt install python3.12-venv -y
```

2. Clone the Repository
```bash
git clone https://github.com/ChenFryd/DL-Assignments.git
cd ./DL-Assignments
```

3. Install the Virtual Environment
```bash
python3 -m venv .venv
```

4. Activate the Virtual Environment

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

5. Install the Requirements
```bash
pip install -r requirements.txt
```