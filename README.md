# UQ in Visual Autoregressive Modeling (VAR) 
Seminar on how to quantify uncertainty in the context of autoregressive visual modeling approaches using conformal predictions and conformalized credal regions (CCR). 

## Sources: 
For the base model architecture please refer to: https://github.com/FoundationVision/VAR
@Article{VAR,
      title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
      author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
      year={2024},
      eprint={2404.02905},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
For the theory behind the approach please refer to: https://arxiv.org/abs/2411.04852

## Structure of the Repo: 
1) Calibration triggered via: calibrating.py
2) Calibration executed in: calibrator.py
3) Functions for generating conformal prediction sets in: models/var.py
