# Description

Python package of utility functions that are useful in analyzing various datasets, in particular, catalogs of particles or galaxies/halos from cosmological simulations.


---
# Requirements

python 3, numpy, scipy, h5py, matplotlib.


---
# Content

## lower-level utilities

### array.py 
* create, manipulate, diagnostics of arrays

### binning.py
* bin array data

### constant.py
* physical constants and unit conversions

### coordinate.py
* manipuate and convert positions and velocities

### io.py
* read, write, print during run time

### math.py 
* math, statistics, and fitting

### plot.py
* supplementary functions for plotting with matplotlib


## higher-level utilities

### catalog.py
* analysis specific to catalogs of halos/galaxies

### cosmology.py
* calculate cosmological values, including cosmic density, distance, age, volume

### halo_property.py
* calculate halo properties at different radii, convert between virial definitions

### orbit.py
* compute orbital quantities such as peri/apo-centric distance, orbital time, given a gravitational potential

### particle.py
* higher-level analysis of N-body particle data

### simulation.py
* tools to help set up and run a simulation


---
# Installing

This package should function either as a subfolder in your `$PYTHONPATH`, or by installing it with `setup.py develop` (which should place an egg.link to the source code in a place that whichever `python` you used to install it knows where to look.


## Instructions for installing as a package

1. create a directory $DIR
2. clone utilities into $DIR
3. copy setup.py from utilities into $DIR
4. run python setup.py develop

In commands, it is:

```
#!bash

DIR=$HOME/code/wetzel_repos/
mkdir -p $DIR
cd $DIR
hg clone https://bitbucket.org/awetzel/utilities.git
cp utilities/setup.py .
python setup.py develop

```

## Instructions for placing in PYTHONPATH

1.  create any directory $DIR
2.  add $DIR to your `$PYTHONPATH`
3.  clone utilities into $DIR

In commands, it is something like:
```
#!bash

DIR=$HOME/code/wetzel
echo $PYTHONPATH=$DIR:$PYTHONPATH >> ~/.bashrc   ### only necessary if not already in your PYTHONPATH
mkdir -p $DIR
cd $DIR
hg clone https://bitbucket.org/awetzel/utilities.git
```

That is, you should end up with `$DIR/utilities/*.py`, with `$DIR` in your `$PYTHONPATH`

You then will be able to import utilities as ut.

To update the repo, cd into $DIR/utilities and run hg pull && hg update.


---
# Licensing

Copyright 2014-2023 by Andrew Wetzel <arwetzel@gmail.com>, Shea Garrison-Kimmel <sheagk@gmail.com>, Jenna Samuel <jsamuel@ucdavis.edu>.

In summary, you are free to use, edit, share, and do whatever you want. But please cite us and report bugs!

Less succinctly, this software is governed by the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
