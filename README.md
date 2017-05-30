# single_cell_reducer

A pipeline for reducing the dimensions of single-cell RNA-seq data.

The expected directory structure is:
single_cell_reducer/
├── data - any input files (usually CSVs), not provided in repo
├── models - output folder where trained models will go
└── reduced_data -output folder where reduced data will go

scrna.py is the main module. Example invocations:

```
python scrna.py -h
```

```
python scrna.py train 2layer_ppitf 100 --sn --gs
```

```
python scrna.py reduce models/2017_05_30-15:23:02_2layer_ppitf --out_folder=reduced_data/test/
```
