# Fine-tuning a Small Language Model in the ReAct Framework

🚧 **Project Status:** In Progress 🚧

## 📌 Project Idea
This project explores how to **fine-tune a Small Language Model (SLM)** using **PyTorch** to operate within the **ReAct (Reason + Act) framework**. The goal is to demonstrate how lightweight transformer models can be adapted into **agentic AI systems** that not only generate text but also **reason, take actions, and use external tools**.

### 🔹 Key Features
- Fine-tuning an SLM with **LoRA adapters** in PyTorch for efficient training.
- Implementing the **ReAct loop** (Thought → Action → Observation → Answer).
- Enabling **tool use** such as:
  - Web/document search
  - Calculator for math reasoning
  - Memory retrieval with embeddings
- Evaluation of reasoning quality, success rate, and tool-use efficiency.

## Getting Started

Follow these steps to set up the project locally.

### Requirements

- **Python version:** 3.10 or above

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/naveenvasou/agentic-sales-pipeline.git
   cd agentic-sales-pipeline
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate the environment**

   - On Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows (PowerShell):
     ```bash
     .venv\Scripts\Activate.ps1
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🤝 Contributions
This is an experimental project in progress. Feedback, suggestions, or collaborations are welcome!
