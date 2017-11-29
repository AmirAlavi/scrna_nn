from setuptools import setup

setup(name='scrna_nn',
      version='0.3.0',
      description='single-cell RNA-seq dimensionality reduction using neural networks',
      author='Amir Alavi',
      packages=['scrna_nn'],
      install_requires=[
          'keras>=2.0.9',
          'docopt',
          'pandas',
          'numpy',
          'matplotlib'
      ],
      scripts=['bin/scrna-nn', 'bin/scrna-nn-data-prep'],
      zip_safe=False)
