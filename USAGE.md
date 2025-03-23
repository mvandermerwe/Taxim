# Taxim Usage

## Calibration

To calibrate, we need to collect data on the target sensor, as described Sec.III.B.c. of the Taxim paper.
Take a small sphere (~4mm) and press it into the sensor surface across the sensor surface. To collect the data:
```bash
python DataCollection/record_Gel.py
```

Next, we manually annotate the contact center and radius for each tactile image. This is done by running:
```bash
python Calibration/generateDataPack.py -data_path DATA_PATH
```
where DATA_PATH is the path where you saved the raw data from the previous step. For each image, use the GUI
to identify the center and radius of the contact for that image.

Finally, we generate the polynomial table used for rendering:
```bash
python Calibration/polyTableCalib.py -data_path DATA_PATH
```
Make sure that the values in `Basics/params.py` and `Basics/sensorParams.py` correspond correctly with your target 
sensor. 

Both `generateDataPack.py` and `polyTableCalib.py` use the pre-processing defined in `Calibration/pre_proc_images.py`
to remove the markers approximately from the image. You may need to tweak some hyperparameters for different sensors, 
or turn this off if markers are not present.

## Optical Simulation

Once we've calibrated our sensor, we can simulate the tactile image formed from a provided depth image. This can be
used to generate a dataset of tactile images and corresponding depths.

To generate simulated depth images similar to an object being grasped with a randomized angle and offset, then small 
perturbations being made to the grasped object, run:
```bash
python OpticalSimulation/generate_sim_depth.py OBJ_CFG DEPTH_OUT_DIR -n NUM -l LENGTH -j JOBS
```

* OBJ_CFG should point to a `yaml` file that specifies the object mesh and sample pose. The sample pose should specify
the frame s.t. the X/Z axes define the plane in which we sample the "grasp" and sample small offsets.
* DEPTH_OUT_DIR is the directory where the generated depth images will be saved.
* NUM is the number of trajectories to sample.
* LENGTH is the length of the trajectory to sample in steps.
* JOBS is the number of parallel jobs to run.

Corresponding tactile images can be generate as follows:
```bash
python OpticalSimulation/simOptical.py DEPTH_OUT_DIR TAC_OUT_DIR -m DATA_PATH -v -j JOBS
```

* DEPTH_OUT_DIR is the directory where the depth images are saved.
* TAC_OUT_DIR is the directory where the tactile images will be saved.
* DATA_PATH is the path to the calibration data generated in the previous step.
* -v is a flag to visualize the simulation.
* -j JOBS is the number of parallel jobs to run.

Make changes as needed to the depth simulation for targeting particular tasks.