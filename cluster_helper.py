import sys
#from utils import utils_imaging
arguments = sys.argv[2:]
command = sys.argv[1]
if type(arguments)== list and len(arguments)>1:
    arguments = ','.join(arguments)
else:
    arguments = arguments[0]
print('{}({})'.format(command,arguments))
