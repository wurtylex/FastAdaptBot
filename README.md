# Fast-Training-Robotics
This repository contains the code for our final project in COMS 4733 ([Course Link](https://robopil.github.io/coms4733/)).

- **Adaptation**: Contains the code for adapting the baseline model online.
- **RL**: Contains the code for training the baseline model.
- **Writeup**: Contains our writeup.
- **main.py**: Provides examples of how to use the various components of the project.
Your mileage may vary if you decide to use this.

Read the writeup for more details!

# Results
Here we have a speed comparison between no adaptation and adaptation.
We observe that an adapted step takes about four times longer, which is still fast enough for rendering, but its performance in a real-world scenario remains uncertain.

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="Results/no adaptation.png" alt="No Adaptation" width="45%" />
  <img src="Results/adaptation.png" alt="Adaptation" width="45%" />
</div>

## 2 legged robot walking 

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="Results/no adapt (2 legs).gif" alt="No Adaptation" width="45%" />
  <img src="Results/adapt (2 legs).gif" alt="Adaptation" width="45%" />
</div>

## 4 legged robot walking 

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="Results/no adapt (4 legs).gif" alt="No Adaptation" width="45%" />
  <img src="Results/adapt (4 legs).gif" alt="Adaptation" width="45%" />
</div>