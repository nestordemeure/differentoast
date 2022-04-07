# DifferenToast

A differentiable implementation of the sky signal simulation capabilities of [Toast](https://github.com/hpc4cmb/toast).

The idea is to build a differentiable function that takes a map of the sky, the positions pointe at by telescopes and simulates the signal observed by the telescope.
Once we have that, we can give it a random sky map, compare the output signal with an actual output signal obtained by a telescope and descend the gradient in order to find a sky map that actually produces the output signal observed.

Limits: 
- This code uses Toast to implement some of its functionality and cannot, currently, be ran without a copy of Toast installed
- This is a proof of concept destined to run on a single node. If you need several node to deal with a large dataset, this will not be useable.

## Components

The code takes a (sky) map and produces a sky signal.

- **scan map**
  take a map and extracts the parts that will be observed by detectors, producing a signal
  map to signal
  used [here](https://github.com/hpc4cmb/toast/blob/707250e6e7e9a5c5497b47ce04faa9b91de6f797/src/toast/scripts/toast_benchmark_satellite#L290-L292)
  inputs defined [here](https://github.com/hpc4cmb/toast/blob/707250e6e7e9a5c5497b47ce04faa9b91de6f797/src/toast/scripts/benchmarking_utilities.py#L683)
  TODO

- **time control convolution**
  represent the fact that, as a detector moves, it is impacted by the values it was previously reading which causes a smoothing
  applied to a signal
  TODO

- **instrumental noise**
  noise introduced by the instruments
  applied to signal
  used [here](https://github.com/hpc4cmb/toast/blob/707250e6e7e9a5c5497b47ce04faa9b91de6f797/src/toast/scripts/toast_benchmark_satellite#L295)
  TODO

- **atmospheric noise**
  noise introduced by the atmosphere (specific to ground telescopes)
  applied to signal
  TODO

## TODO

The plan is to go from a very simple simulator, see how well we can reverse it then add components of increased realism until we have a fully realistic simulator.
Working on satellites first then ground telescopes (as those need to also take things like atmospheric noise into account).

Once the simulator is fully realistic and if we could reverse intermediate steps then we could try and reverse actual telescope data and see what happens.

- find out exactly what we need for our functions
- import/generate the data using toast
- export it from the toast format into a very easy to parse format that contains only the information we need (and no MPI reference)

- extract scan map and get it running within this repository for some realistic values
- make the inputs and outputs of scanmap clear
- make scan map differentiable
- try and reverse scanmap

- add instrumental noise
- add time control convolution
- add atmospheric noise
