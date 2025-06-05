# UQ in Visual Autoregressive Modeling (VAR) 
Seminar on how to quantify uncertainty in the context of autoregressive visual modeling approaches using conformal predictions and conformalized credal regions (CCR). 

## Sources: 
For the base model architecture please refer to: 
https://github.com/FoundationVision/VAR
```
@Article{VAR,
      title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
      author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
      year={2024},
      eprint={2404.02905},
      archivePrefix={arXiv},
      primaryClass={cs.CV}}
```

For the theory behind conformalized credal regions please refer to: 
https://arxiv.org/abs/2411.04852

## Structure of the Repo: 
1) Calibration triggered via: `calibrating.py`
      - Different types of calibration techniques can be submitted as calibration parameter
3) Calibration executed in: `calibrator.py`
4) Functions for generating conformal prediction sets in: `models/var.py`
      - Specific function for generating images with CPS/CCR (depending on the type of calibration of the calibrated model): `autoregressive_infer_and_uq()`
5) Helper functions for e.g. calculating the upper/lower entropy can be found in `models/helpers_calib.py`
6) Example on how to use a calibrated model in `demo_calibrate.ipynb`

Newly created/modified files in the context of this seminar: `calibrating.py`, `calibrator.py`, `models/var.py`, `models/vqvae.py`, `models/quant.py`, `models/__init__.py`, `demo_calibrate.ipynb`, `utils/args_util.py`, `utils/data.py`
