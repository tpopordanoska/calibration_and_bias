# Calibration and Bias


This is the code that accompanies the paper "On the relationship between calibrated predictors and unbiased volume estimation" (accepted as a conference paper at MICCAI 2021).


## Dependencies

If you use [Poetry](https://python-poetry.org/), navigate to the project root directory and run `poetry install`. 


## Experiments
To run the experiments:

1. Train the models and obtain predictions as described in https://github.com/AxelJanRousseau/PostTrainCalibration.
2. Place the predictions in appropriate subfolders, i.e. .data/predictions/BRATS_2018 and .data/predictions/ISLES2018.
3. Run `main.py` to save all plots and tables.


## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).


## Acknowledgements

We acknowledge funding from the Flemish Government under the Onderzoeksprogramma Artificiele Intelligentie (AI) Vlaanderen programme
