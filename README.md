# single_cell_reducer

A pipeline for reducing the dimensions of single-cell RNA-seq data.

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
