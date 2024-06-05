# ultranest-pitchfork
emulating individual modes of solar like oscillators for use in fundamental parameter posterior sampling using ultranest

## structure
.\
├── form\
│   ├── hare0\
│   │   ├── hare0.json\
│   │   ├── obs0\
│   │   │   ├── obs0.json\
│   │   │   ├── obs0-samples.json\
│   │   │   ├── obs0-posterior.png\
│   │   │   ├── obs0-posterior-predictive.png\
│   │   │   └── ...\
│   │   ├── ...\
│   │   └── obs5\
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
└── README.md
