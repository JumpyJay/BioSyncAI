# PulseFlow AI: Closed-Loop Biofeedback for Flow State Maintenance

[cite_start]PulseFlow AI is a real-time system designed to combat the "Fragmented Attention" and "Dopamine Disparity" caused by modern social media algorithms[cite: 10, 13]. [cite_start]By utilizing heart rate variability (HRV) as a proxy for the Central Autonomic Network (CAN), the system detects cognitive states and uses generative music as a regulatory tool to guide users into an optimal "Flow State"[cite: 19, 41].

## 🧠 Project Overview
[cite_start]The core objective of this project is to answer: *Can a real-time AI system use physiological biomarkers to detect and regulate a user's cognitive state?*[cite: 19]. [cite_start]Rather than treating music as static background noise, this system treats it as a dynamic, closed-loop biofeedback signal[cite: 20].

### Theoretical Framework
* [cite_start]**Flow Theory:** Based on Csikszentmihalyi’s definition of flow as a state of complete absorption in the present moment[cite: 23].
* [cite_start]**Neurovisceral Integration:** Based on Thayer and Lane’s model, where heart rate variability (HRV) reflects the functional integrity of the Central Autonomic Network (CAN)[cite: 41].
* [cite_start]**The Nine-Quadrant Model:** The system classifies user states into categories such as Anxiety, Arousal, Flow, Control, Relaxation, Boredom, Apathy, and Worry[cite: 24].

## 🛠️ Technical Architecture

### 1. Data Acquisition (The Input)
[cite_start]The system extracts real-time physiological data through a wearable HRV sensor[cite: 68]. Key features include:
* [cite_start]**BPM** (Beats Per Minute)[cite: 71].
* [cite_start]**RMSSD** (Root Mean Square of Successive Differences)[cite: 71].
* [cite_start]**HR Delta** (Heart Rate variability trends)[cite: 71].

### 2. Intelligence Layer (SVM Classification)
[cite_start]The system utilizes a **Multidimensional Multiclass Support Vector Machine (SVM)** to predict cognitive states[cite: 56, 58].
* [cite_start]**Kernel Trick:** Projects biomarkers into higher-dimensional space to handle non-linear physiological data[cite: 64, 74].
* [cite_start]**Feature Scaling:** Standardizes data to ensure the hyperplane is not biased toward dimensions with larger numerical ranges[cite: 65, 66].
* [cite_start]**Pre-training:** The model is pre-trained on the **WESAD dataset** to establish baseline stress vs. relaxation boundaries[cite: 72].
* [cite_start]**Personalization:** Conducts fine-tuning by collecting user-specific data during tasks and labeling them via a Flow State Scale assessment[cite: 75, 76].

### 3. Generative Music Engine (Reinforcement Learning)
[cite_start]Rather than simple rule-based mapping, the system uses a **Reinforcement Learning (RL)** framework[cite: 80].
* [cite_start]**Modulation:** Adjusts parameters such as tempo, rhythmic density, harmonic complexity, and spectral brightness[cite: 79].
* [cite_start]**Reward Signal:** The stabilization of HRV balance serves as the intrinsic reward, ensuring the user moves toward the Flow quadrant[cite: 82].

## 🚀 Getting Started

### Prerequisites
* Raspberry Pi (with OpenClaw/Python 3.x)
* Pulse Sensor (Analog/Digital)
* `scikit-learn` for SVM implementation
* `numpy` & `scipy` for signal processing

### Implementation Notes
* [cite_start]**Edge Cases:** The system includes personalized calibration to handle baseline distributions and avoid misclassifying pathological irregularities like arrhythmia[cite: 84, 85].
* [cite_start]**Deployment:** The user's normalized physiological input is fed into the real-time trained SVM classifier to predict current cognitive states[cite: 78].

## 📚 References
* Csikszentmihalyi, M. (1997). *Finding flow: The psychology of engagement with everyday life*.
* Thayer, J. F., & Lane, R. D. (2000). *A model of neurovisceral integration in emotion regulation and dysregulation*.
* Satani, A., et al. (2025). *Modern day high: The neurocognitive impact of social media usage*.
