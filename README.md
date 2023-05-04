# ColorfulCurves

*ColorfulCurves*: Palette-Aware Lightness Control and Color Editing via Sparse Optimization

*ACM Transactions on Graphics (TOG)*. Presented at *SIGGRAPH North America 2023*.

[*By* [Cheng-Kang Ted Chao](https://mason.gmu.edu/~cchao8/), Jason Klein, [Jianchao Tan](https://scholar.google.com/citations?user=1Gywy80AAAAJ&hl=en), [Jose Echevarria](http://www.jiechevarria.com/), [Yotam Gingold](https://cragl.cs.gmu.edu/)] 


## About

This repo is official code release for *ColorfulCurves*.

## Installation

Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
(Miniconda is faster to install.) Choose the 64-bit Python 3.x version. Launch the Anaconda shell from the Start menu and navigate to the posterization directory.
Then:

    conda env create -f environment.yml
    conda activate sparse_edit

## Usage

Launch the GUI:

    python GUI.py
    
Note: The time complexity of our algorithm is **independent** to image size; however, the GUI will crop your image if its width larger than certain size to fit on a screen. Our GUI is not implemented in a way that it can dynamically shrink or enlarge your image when changing window size.


## License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

The software is licensed under `MIT License`.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
