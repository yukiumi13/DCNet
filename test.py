import getopt
import os
import sys
opts, args = getopt.getopt(sys.argv[1:], "h", ["ifile=", "ofile="])
print('opts',opts)
print('args',args)
