# EEG AI MODEL POWERED BUGS Project

A notebook LM Deepdive with the bugs in action video at youtube

https://www.youtube.com/watch?v=_QaS5uLAeWI

## Overview

The **EEG BUGS Project** is a comprehensive toolkit designed for analyzing EEG (Electroencephalography) data and visualizing intelligent simulations based on neural insights. This project comprises two main components:

1. **EEG AI Model Maker:** A tool for training an EEG autoencoder model using `.edf` files.
2. **EEG Bug Simulator:** An interactive simulation where intelligent bugs navigate environments influenced by EEG-derived data.

## Features

### EEG AI Model Maker
- **Train EEG Autoencoder:** Process EEG `.edf` files to train a convolutional autoencoder model.
- **Configuration Options:** Customize training parameters such as epoch length, batch size, number of epochs, and learning rate.
- **Progress Monitoring:** Real-time training progress with logs and checkpointing.
- **Save Models:** When the model is done download the pth model and then when you launch the EEG bugs software choose that as eeg model. (Hidden vectors file is also made)

### EEG Bug Simulator
- **Intelligent Bugs:** Simulate bugs with configurable wave neurons that interact based on EEG data.
- **Dynamic Vision:** Bugs have a defined vision cone to detect other bugs and environmental inputs.
- **Communication:** Bugs share their internal states through ASCII-based messages derived from neural activations.
- **Drawing Capability:** Visual trails and echo traces represent bug interactions and behaviors.
- **Flexible Input Sources:** Choose between webcam feeds or static background images for simulation environments.

## Installation

1. **Clone the Repository:**

    git clone https://github.com/anttiluode/eegbugs.git

    cd eegbugs
   
3. **Install Dependencies:**
 
    pip install -r requirements.txt
    
    *`requirements.txt` should include:*

    numpy
    torch
    mne
    scipy
    tqdm
    matplotlib
    pillow
    opencv-python
    
## Usage

### 1. Train the EEG Autoencoder Model

Before running the simulator, you need to train the EEG autoencoder model using the **EEG AI Model Maker**.

   cd EEGAIModelMaker
   
2. **Run the Model Maker:**
 
    python eeg_ai_model_maker.py
   
    
4. **Configure and Train:**

    - Find .edf ending EEG file from online. 
    - **Load EDF File:** Click the **Browse** button to select your `.edf` EEG data file.
    - **Set Parameters:** Adjust training parameters as needed (epoch length, batch size, etc.).
    - **Start Training:** Click the **Start Training** button to begin. Monitor progress through the interface.
    - **Completion:** Upon training completion, the model (`eeg_autoencoder_model.pth`) and hidden vectors (`hidden_vectors.npy`) can be downloaded. 


### 2. Run the EEG Bug Simulator

With the trained model, you can now launch the **EEG Bug Simulator** to visualize intelligent bug interactions.

    python app.py

3. **Configure and Start:**

    - **Select EEG Model:** In the **Configuration** tab, click **Browse** to select the trained `.pth` model from the `features` directory.
    - **Choose Input Source:** Select between using a webcam or uploading a background image.
    - **Configure Wave Neurons:** Set the number of wave neurons per bug to influence their behavior.
    - **Start Simulation:** Click the **Start Simulation** button to begin. Observe bugs interacting in real-time within the simulation canvas and discussion sidebar.
    - Notice the simulation runs in its own tab.
    - 
## License

This project is licensed under the [MIT License](LICENSE).

**Enjoy exploring the fascinating dynamics of EEG-driven intelligent simulations!** 🧠✨

