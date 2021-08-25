# Calibration and Bias


This is the code that accompanies the paper "On the relationship between calibrated predictors and unbiased volume estimation" (accepted as a conference paper at MICCAI 2021). The full text of the paper can be found here: https://lirias.kuleuven.be/handle/123456789/677210


## Dependencies

If you use [Poetry](https://python-poetry.org/), navigate to the project root directory and run `poetry install`. 


## Experiments
To run the experiments:

1. Train the models and obtain predictions as described in https://github.com/AxelJanRousseau/PostTrainCalibration.
2. Place the predictions in appropriate subfolders, i.e. .data/predictions/BRATS_2018 and .data/predictions/ISLES_2018.
3. Run `main.py` to save all plots and tables.

## Reference

If you found this code useful, please cite:

```
@incollection{Popordanoska2021a,
year={2021},
booktitle={Medical Image Computing and Computer-Assisted Intervention},
title={On the relationship between calibrated predictors and unbiased volume estimation},
author={Teodora Popordanoska and Jeroen Bertels and Dirk Vandermeulen and Frederik Maes and Blaschko, Matthew B.},
}
```

```
@inproceedings{Rousseau2020a,
  title={Post Training Uncertainty Calibration of Deep Networks for Medical Image Segmentation},
  AUTHOR = {Rousseau, Axel-Jan and Thijs Becker and Jeroen Bertels and Matthew B. Blaschko and Dirk Valkenborg},
  YEAR = {2021},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
}
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).


## Acknowledgements

We acknowledge funding from the Flemish Government under the Onderzoeksprogramma Artificiele Intelligentie (AI) Vlaanderen programme.
