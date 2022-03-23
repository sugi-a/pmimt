import sys
from transformer.vanilla.train import main

argv = ['--accums', '6', '--debug']

main(argv + sys.argv[1:])
