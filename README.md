# NetTCR in PyTorch 
This repository offers a PyTorch implementation of [NetTCR-2.2](https://github.com/mnielLab/NetTCR-2.2), a deep learning model used to predict TCR specificity. 

Currently only training and testing in mode `pan` is supported. 

Functions supported by this repo include:
- training a PyTorch model for NetTCR binding prediction given peptide sequence
- converting a tensorflow .tflite file (you get get them from NetTCR-2.2 repo) to .onnx file which can be read and used in PyTorch
- make predictions for peptide binding in PyTorch

Important files:
- `src/nettcr_archs.py`: network architecture file 
- `train_pan`: code for nettcr prediction model training
- `eval.py`: code for making peptide binding prediction 
- `model_eval.ipynb`: notebook for reading .onnx file and making peptide binding prediction 

## Citation
Mathias Fynbo Jensen, Morten Nielsen (2024) **Enhancing TCR specificity predictions by combined pan- and peptide-specific training, loss-scaling, and sequence similarity integration** *eLife* **12**:RP93934. https://doi.org/10.7554/eLife.93934.3