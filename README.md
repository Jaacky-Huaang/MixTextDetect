# MixTextDetect: Advancing the Detection of Mixed Human-Written and Machine-Generated Text

**NYUSH Fall 24 CSDSE Capstone Project**  
**Authors**: Junjie Huang, Lihan Feng, Yiling Cao  

---

## Overview

The increasing prevalence of Large Language Models (LLMs) has introduced significant challenges in identifying mixed-text content, where human-written, machine-generated, and collaboratively refined texts coexist. Existing detection methods, primarily designed for binary classification, fail to address this complexity. To bridge this gap, our study proposes a three-class classification framework to accurately distinguish these text categories. Leveraging metric- and model-based approaches, we introduced a brute-force sampling strategy based on text length distributions and a more
efficient strategy based on entropy loss, with the latter reducing computational costs by over 60-fold compared to the former. Our results demonstrate the efficacy of encoder-only models, with optimized sampling strategies improving F1 scores by 0.07 (achieving 0.92), and highlight the superior performance of decoder-only models, which attained an F1 score of 0.98 through instruction tuning.

---

## File Structure

### Main Files
- **`baseline.py`**: Contains the baseline implementation of the text detection model.
- **`dataset_loader.py`**: Handles loading and preprocessing of datasets for training and evaluation.
- **`entropy.py`**: Implements entropy-based methods for text analysis and detection.
- **`grid_search.py`**: Performs hyperparameter optimization for the models using grid search.
- **`train_random.py`**: Trains models with random initialization.
- **`train_transfer.py`**: Trains models using transfer learning techniques.

### Folders
- **`methods/`**: Includes various utility functions and model-specific methods for training and evaluation.
- **`plot/`**: Contains scripts for visualizing results, metrics, and other aspects of the project.

### Other Files
- **`requirements.txt`**: Lists the required Python libraries and dependencies for the project.
- **`qwen-7b_lora_sft.yaml`**: Contains the configuration for fine-tuning the qwen2.5-7b model.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jaacky-Huaang/MixTextDetect.git
   cd MixTextDetect
    ```
2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. To run baseline method:
    ```bash
    python baseline.py
    ```
4. To run grid search method:
    ```bash
    python grid_search.py
    ```
5. To run entropy ranking method:
    ```bash
    python entropy.py
    ```
6. To fine-tune a qwen2.5-7b model, we recommend using LLaMA-Factory (installed following this link: https://github.com/hiyouga/LLaMA-Factory.git) and the provided configuration file `qwen-7b_lora_sft.yaml`.

