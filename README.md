# ColorfulCurves

*ColorfulCurves*: Palette-Aware Lightness Control and Color Editing via Sparse Optimization ([paper]([https://cragl.cs.gmu.edu/colorfulcurves/ColorfulCurves-%20Palette-Aware%20Lightness%20Control%20and%20Color%20Editing%20via%20Sparse%20Optimization%20(Cheng-Kang%20Ted%20Chao,%20Jason%20Klein,%20Jianchao%20Tan,%20Jose%20Echevarria,%20Yotam%20Gingold%202023%20SIGGRAPH)%20small.pdf])

*ACM Transactions on Graphics (TOG)*. Presented at *SIGGRAPH North America 2023*.

[*By* [Cheng-Kang Ted Chao](https://mason.gmu.edu/~cchao8/), [Jason Klein](https://www.linkedin.com/in/jason-adam-klein), [Jianchao Tan](https://scholar.google.com/citations?user=1Gywy80AAAAJ&hl=en), [Jose Echevarria](http://www.jiechevarria.com/), [Yotam Gingold](https://cragl.cs.gmu.edu/)] 

See demo video in our [project page](https://cragl.cs.gmu.edu/colorfulcurves/) for our editing framework.

## About

This repo is official code release for *ColorfulCurves*. 

Our editing framework allows users to:
1. Edit lightness (a.k.a luminance in photography communities though not mathematically equivalent) sparsely according to the representative colors in the given image.
2. Edit pixel's colors by directly placing image-space color constraints. *ColorfulCurves* will find a best set of representative colors and tone curves to satisfy your constraints. 

## Installation

You can install dependencies using either `conda` or `pip`.

### Conda

Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
(Miniconda is faster to install.) Choose the 64-bit Python 3.x version. Launch the Anaconda shell from the Start menu and navigate to this directory.
Then:

    conda env create -f environment.yml
    conda activate colorfulcurves

To update an already created environment if the `environment.yml` file changes, first activate and then run `conda env update --file environment.yml --prune`.

### Pip

(Optional) Create a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

(untested) If you want to install the exact version of the dependencies we used, run: `pip install -r requirements.frozen.txt`

### (Optional) Compile Cython

After setting up the environment, you can compile the Cython file with `cythonize -i func/aux/GteDistPointTriangle.pyx`. If you don't, it will happen automatically when you launch the GUI.


## Usage

Launch the GUI:

    python3 GUI.py

Note: The time complexity of our algorithm is **independent** of image size; however, the GUI will resize your image if its width is larger than a certain size to fit itself properly onscreen. A better implementation would operate on the full size image and just show a downsampled version.


## License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

The software is licensed under `MIT License`.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
