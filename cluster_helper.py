import sys
from utils import utils_imaging
print(sys.argv)
arguments = sys.argv[2:]
command = sys.argv[1]
if command == 'utils_imaging.register_trial':
    arguments_real = list()
    for argument in arguments:
        arguments_real.append('"'+argument+'"')
    arguments=arguments_real
if type(arguments)== list and len(arguments)>1:
    arguments = ','.join(arguments)
else:
    arguments = arguments[0]
print(command)
print(arguments)
eval('{}({})'.format(command,arguments))