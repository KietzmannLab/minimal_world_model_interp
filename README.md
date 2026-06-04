# Mechanisms of sequential world modeling
Official codebase for the paper: 
#Path Integration and Object-Location Binding Emerge in an Action-Conditioned Predictive Sequence Network
**Linda Ariel Ventura, Victoria Bosch, Tim C Kietzmann, Sushrut Thorat**

This repository is geared towards making all the code and analyses from the [🔗 paper](https://arxiv.org/abs/2602.03490) possible.  

### Abstract
*Adaptive cognition requires structured internal models of objects and their relations. Predictive neural networks are often proposed to learn such world models, but how these are instantiated and how they support prediction remain unclear. We investigate this in a minimal in-silico setting. A recurrent neural network samples tokens sequentially from 2D continuous token scenes and is trained to predict the upcoming token from the current input and a saccade-like displacement. On novel scenes, prediction accuracy improves across the sequence, indicating in-context learning. Decoding analyses reveal path integration and dynamic binding of token identity to position. Interventional analyses show that new bindings can be learned late in sequence and that out-of-distribution bindings can be learned as well. Together, these findings show how structured representations relying on flexible binding emerge to support prediction, offering a mechanistic account of sequential world modeling relevant to cognitive science.*


### Codebase
1. The figures included in the paper can be plotted using [paper_plots](paper_plots.ipynb)  
2. The training of the network can be done with [train_model](setup/train_model.py)  
3. Scripts to generate the token sequences can be found under [sequence_maker](setup/sequence_maker.py)  
4. Scripts to train the SVMs for decoding analyses can be done using [run_decoders](decoding_analysis/run_decoders.ipynb)  

### Requirements  

### Environment setup  
The code has been tested with:  
- Python 3.13.5 + PyTorch 2.10.0 (CPU) on macOS (Apple M1)  
- Python 3.13.5 + PyTorch 2.10.0 on a Linux HPC cluster  

Clone this repository (or download it directly):  

```bash  
git clone https://github.com/KietzmannLab/simple_gpn_interpretability.git  
cd simple_gpn_interpretability  
```

Setup your enviroment as:

```bash
conda env create -f environment.yml
conda activate interpretability
```

Uncomment lines 30-32 from environment.yml only if training is intended. For running analysis those packages are not necessary.  

### Pretrained networks and analysis data  
1. The model, SVM results, and analysis sequences used in the paper are available under [paper_data](paper_data).  

## Citation

If you use these networks or any of the provided analyses in your research, please cite our paper:
```
Ventura, L. A., Bosch, V., Kietzmann, T. C., & Thorat, S. (2026). A Minimal Task Reveals Emergent Path Integration and Object-Location Binding in a Predictive Sequence Model. arXiv preprint arXiv:2602.03490.
```
