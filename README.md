# MPC and Lyapunov for Drones

## Overview
This project explores the application of Model Predictive Control (MPC) and Lyapunov-based methods for controlling drones.


## Installation
1. Clone the repository.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Ensure you have Python 3.8+ and necessary libraries like MuJoCo, NumPy, CasADi, and Matplotlib.

## File Layout
The `hybrid_mpc` subdirectory contains the most up-to-date version of the controller and certificate pipeline. The other subdirectories contain other experiments, cached models, or results. The main subdirectory contains the XML file defining the drone and the data collection scripts.

## Usage
All the necessary models and rollout cache files have been provided. In order to regenerate the plots run the following commands:  
 ```
 cd hybrid_mpc
 python mpc_hybrid.py
 ```

This will run the hybrid mpc controller and will start a MuJoCo visualization of the controller in action. It will also save a PNG file called hybrid_mpc.png with the tracking performance.

To generate the benchmark plot run the following from the main directory:  
```
cd hybrid_mpc
python benchmark_hybrid.py
```

This will generate benchmark_hybrid_summary.png and benchmark_hybrid_tracking.png files with the benchmark results. 

To generate the safety certificate, run the following from the main directory. _Please note that this took over 3 days to run on our machine. If you would like to view the results, they are found in the hybrid_mpc/certificate_outputs subdirectory._
```
cd hybrid_mpc
python lyapunov.py
```

This will store the outputs in the `hybrid_mpc/certificate_outputs` subdirectory.
