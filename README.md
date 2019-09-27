# Chrononet

Dependencies:
    numpy
    Python 3.6+
    ELMo
    Pytorch 1.0.1 (with CUDA support if you want to use a GPU)
    sklearn-crfsuite (for CRF sequence tagger)

Setup:

    ./setupy.py
    pip install .


## Usage:

1. Convert data to a csv file according to the header in data_tools/data_scheme
OR - extend data_adapter.py and create your own adapter to load and write to your desired data format.

2. Create a config file including parameters for the models you want to run. See config_example_* for examples. 

    Stage options include:

    `sequence` (time and event tagging using BIO labels): model options: `crf`

    `ordering` (listwise temporal ordering): model options: `neurallinear` (RNN), `neural` (Set2Seq)

    `classification`
    
3. `python chrononet.py --config [CONFIG].ini`

## Temporal ordering models

Temporal ordering models from the 2019 paper (see below) are in models/ordering/pytorch_models.py . The 'linear' model is called GRU_GRU, and the set2sequence model is called Set2Sequence.

### Citation

If you use this code, please cite the following paper:

Serena Jeblee, Frank Rudzicz, Graeme Hirst. 2019. Neural temporal ordering of events in medical narratives with a set-to-sequence model. In <i>Proceedings of Machine Learning for Health Workshop (ML4H 2019)</i>. (In press). 

Previous papers
    
Serena Jeblee, Graeme Hirst. 2018. <a href="http://www.cs.toronto.edu/~sjeblee/files/LouhiPaper47cameraready.pdf">Listwise temporal ordering of events in clinical notes</a>. In <i>Proceedings of the Ninth International Workshop on Health Text Mining (LOUHI 2018)</i>. (In press). 
    
    
