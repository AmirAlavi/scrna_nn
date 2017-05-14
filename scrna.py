"""Single-cell RNA-seq Analysis Pipeline.

Usage:
  scrna.py train <architecture>
  scrna.py (-h | --help)
  scrna.py --version

Options:
  -h --help           Show this screen.
  --version           Show version.
  --epochs=<nepochs>  Number of epochs to train for [default: 20]. Only used
                      for training neural network architectures.

"""
from docopt import docopt

from util import ScrnaException

if __name__ == '__main__':
        args = docopt(__doc__, version='scrna 0.1')
        # print(args)
        try:
          if args['train']:
              print("Training")
              raise ScrnaException("Example exception")
        except ScrnaException as e:
            msg = e.args[0]
            print("scrna excption: ", msg)