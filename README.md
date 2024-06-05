# ultranest-pitchfork
emulating individual modes of solar like oscillators for use in fundamental parameter posterior sampling using ultranest

## structure
.\
├── form\
│   ├── hare0\
│   │   ├── hare0.json\
│   │   ├── hare0-obs0\
│   │   │   ├── hare0-obs0.json\
│   │   │   ├── hare0-obs0-samples.json\
│   │   │   ├── hare0-obs0-posterior.png\
│   │   │   ├── hare0-obs0-posterior-predictive.png\
│   │   │   └── ...\
│   │   ├── ...\
│   │   └── hare0-obs5\
│   ├── ...\
│   └── hare100\
├── nest\
│   ├── emu0\
│   ├── ...\
│   └── emu100\
├── pitchfork\
│   ├── pitchfork.h5\
│   ├── pitchfork-cov.json\
│   └── pitchfork-info.json\
├── scripts\
│   ├── InversePCA.py\
│   ├── WMSE.py\
│   ├── ultranest-sampler.py\
│   └── ...\
├── LICENSE\
└── README.md\
