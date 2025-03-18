# Bert Fine Tuning

This repository contains two projects involving the fine-tuning of BERT. The first project focuses on **slot filling** and **intent classification** using the **ATIS dataset**. The second project is focused on **aspect extraction** for **aspect-based sentiment analysis** using the **Laptop partition of SemEval2014 Task 4 dataset**.

---

## Getting Started

To set up the environment and install the necessary dependencies, follow these steps:

1. Create and activate the conda environment:

    ```bash
    conda create -n bert-finetune python=3.10.13
    conda activate bert-finetune
    ```

2. Install the required dependencies from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

3. To run one of the projects, navigate to the desired project folder and execute:

    ```bash
    python main.py
    ```