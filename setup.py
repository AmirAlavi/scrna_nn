from setuptools import setup

setup(name='scrna_nn',
      version='0.3.0',
      description='single-cell RNA-seq dimensionality reduction using neural networks',
      author='Amir Alavi',
      packages=['scrna_nn'],
      install_requires=[
          'keras>=2.1.6',
          'docopt',
          'pandas',
          'tables',
          'numpy',
          'matplotlib',
          'scikit-learn',
          'pydot',
          'graphviz',
          'h5py',
          'imageio',
          'sparsely-connected-keras',
          'tied-autoencoder-keras>=0.4.0'
      ],
      python_requires='>=3',
      scripts=['bin/scrna-nn', 'bin/scrna-nn-data-prep', 'bin/scrna-nn-impute'],
      #zip_safe=False
)
