# =============================================================================
# ## ephysIO python module
# ##
# ## Reads the following data formats into python:
# ## - uncompressed ACQ4 HDF5 binary recording files (.ma)
# ## - ephysIO HDF5-based MATLAB 7.3 files (.mat)
# ##
# ## Writes the following data formats from python:
# ## - ephysIO HDF5-based MATLAB 7.3 files (.mat)
# ##
# ## version 29 April 2018
# ## To do:
# ##   - add module, function and file format info and help
# ##   - add automatic data rescaling where units have a prefix (e.g. u'\xb5')
# ##   - add check for wave names and apply renaming to comply with name rules
# 
# def MATload(filepath):
#     
#     """
#     Load data from ephysIO HDF5 files into python
#     """
#     
#     # Check filetype
#     fid = open(filepath, 'r')
#     matver = fid.read(10)
#     if matver != 'MATLAB 7.3':
#         raise ValueError("File is not Matlab HDF5 (v7.3)")
#     
#     # Create empty data dictionary
#     data={}
# 
#     # Import numpy for numeric data handling
#     import numpy as np
#             
#     # Load File
#     import hdf5storage
#     mat = hdf5storage.loadmat(filepath)
# 
#     # Convert data types
#     data['array'] = np.array(mat.pop('array'),dtype='float64')
#     data['xunit'] = str(mat.pop('xunit')[0][0])
#     data['yunit'] = str(mat.pop('yunit')[0][0])
#     data['xdiff'] = mat.pop('xdiff')[0][0].astype('float64')
#     scale = mat.pop('scale').astype('float64')
#     start = mat.pop('start').astype('float64')
#     n = mat.get('names').size
#     data['names'] = [str(mat.get('names')[i][0]) for i in range(n)]
#     data['notes'] = [str(mat.get('notes')[i][0]) for i in range(mat.get('notes').size)]
#     if 'saved' in mat:
#         from datetime import datetime, timedelta
#         saved = mat.pop('saved')[0][0].astype('float64')
#         saved = datetime.fromordinal(int(saved)) + timedelta(days=saved%1) - timedelta(days = 366)
#         from re import sub
#         data['saved'] = sub(r'\W+','',saved.isoformat())
#     else:
#         data['saved'] = ''
# 
#     # Calculate power of 2 scale factor
#     scale = 2.0**scale.astype('float64')
# 
#     # Rescale transformed data array
#     for i in range(n):
#         data['array'][i] = data.get('array')[i]/scale[i]
# 
#     # Backtransform data array to real world values
#     data['array'] = np.concatenate((start,data.pop('array')),1).cumsum(1)
# 
#     # Calculate X dimension for constant sampling interval
#     if data.get('xdiff') > 0:
#         x = data.get('xdiff') * np.arange(0.0,np.shape(data.get('array'))[1],1,'float64')
#         data['array'] = np.concatenate((np.array(x,ndmin=2),data.pop('array')),0)
#         data['names'] = [str(mat.pop('xname')[0][0])] + data.pop('names')
#     
#     return data
# 
# 
# def MATsave(filepath, array, xunit, yunit, names = None, notes = None):
#     
#     """
#     Save n-dimensional data array and properties to ephysIO HDF5 files.
#     Data must be formatted in array such that the units are without prefix.
#     """
# 
#     # Import numpy for numeric data handling
#     import numpy as np
#     
#     # Transform data array by representing it as difference values
#     start = np.array(np.mat(array.T[0]).T)
#     array = np.diff(array,1,1)
# 
#     # Check sampling properties of the X dimension
#     if np.any(np.abs(np.diff(array[0]))>1.192093e-7):
#         # Variable sampling interval
#         xdiff = 0.0
#     else:
#         # Constant sampling interval
#         if np.abs(start[0][0])>0:
#             print('Note: Timebase offset will be reset to zero')
#         xdiff = array[0][0]
#         array = array[1::]
#         start = start[1::]
#     
#     # Scale each element of the transformed data array by a power-of-2 scaling factor
#     maxval = np.max(np.abs(array),1)
#     scale = np.array(np.mat(np.fix(np.log2(32767.0/maxval))).T)
#     n = len(array)
#     for i in range(n):
#             array[i] = array[i]* 2.0**(scale[i])
# 
#     # Change class of variables for more efficient data storage
#     start = start.astype('float32')
#     scale = scale.astype('uint8')
#     array = np.rint(array).astype('int16')
# 
#     # Copy data to python dictionary
#     data = {}
#     data['array'] = array
#     data['start'] = start
#     data['scale'] = scale
#     data['xdiff'] = np.array([[xdiff]])
#     data['xunit'] = xunit
#     data['yunit'] = yunit
# 
#     # Use customized names
#     if names != None:
#         if xdiff>0:
#             data['xname'] = np.array([[names[0]]])
#             names = names[1::]
#             data['names'] = [[names[i]] for i in range(n)]
#         elif xdiff==0:
#             data['xname'] = np.array([['']])
#             data['names'] = [[names[i]] for i in range(n)]
#     else:
#         if xdiff>0:
#             if xunit=='s':
#                 data['xname'] = np.array([['Time']])
#             else:
#                 data['xname'] = np.array([['XWave']])
#             data['names'] = [['YWave%d' %i] for i in range(n)]
#         elif xdiff==0:
#             data['xname'] = np.array([['']])
#             if xunit=='s':
#                 data['names'] = [['Time']]
#             else:
#                 data['names'] = [['XWave']]
#             [[data['names'].append(['YWave%d' %i])] for i in range(n-1)]
#     data['names'] = np.array(np.mat(data.get('names')))  
# 
#     # Add notes
#     if notes != None:
#         try:
#             notes = [[notes[i]] for i in range(len(notes))]
#             data['notes'] = np.array(np.mat(notes))
#         except:
#             pass
#     else:
#         data['notes'] = np.array([['']])
# 
#     # Create variable recording serial number of date and time
#     from datetime import datetime, timedelta
#     t = datetime.now()
#     data['saved'] = np.array([[(t+timedelta(days=366.0)).toordinal() +
#                     (t-datetime(t.year,t.month,t.day,0,0,0)).seconds/86400.0]])
# 
#     # Save data
#     import hdf5storage
#     hdf5storage.savemat(filepath,data,appendmat=True,format='7.3',matlab_compatible=True)
# 
#     return
# 
# def MAload(filepath, ch=1):
#     
# """
# Load electrophysiology recording data from the primary recording 
# channel of acq4 hdf5 (.ma) files.
# 
# If the file is in a folder entitled 000, load acq4 will load
# the recording traces from all sibling folders (000,001,002,...)
# """
# 
# =============================================================================
# Move to file directory and load File
import os
import h5py
if '/' in filepath:
    pass
else:
    filepath = './'+filepath
os.chdir(filepath.rsplit('/',1)[0])
h5 = h5py.File(filepath,'r')
filename = filepath.rsplit('/',1)[1]

# Pass metadata into data dictionary
import numpy as np
data = {}
metadata = h5.get('info')
data['array'] = [metadata.get('/info/1/').get('values')[:]]
data['xdiff'] = data.get('array')[0][1]
data['xunit'] = metadata.get('/info/1/').attrs.get('units')[1:-1]
data['yunit'] = metadata.get('/info/0/cols/%s' %(ch)).attrs.get('units')[1:-1]

# Pass data into the array
data['names'] = ['Time']
if os.getcwd()[-3::] == '000':
    os.chdir('..')
    count = 0
    exitflag = 0
    while exitflag < 1:
        dirname = '00'+str(count)
        dirname = dirname[-3::]
        if os.path.isdir(dirname):
            data['names'].append('YWave%s' % dirname)
            os.chdir(dirname)
            if os.path.isfile(filename): 
                h5 = h5py.File(filename,'r')
                data['array'].append(h5.get('data')[ch].tolist())
            else:
                exitflag = 1
                #raise ValueError("The file '%s' is missing from sibling directory '%s'" % (filename,dirname))
            count+=1
            os.chdir('..')
        else:
            exitflag = 1
else:
    data['names'].append('YWave%s' % os.getcwd()[-3::])
    h5 = h5py.File(filename,'r')
    data['array'].append(h5.get('data')[ch].tolist())
data['array'] = np.array(data.pop('array'))
    
# Parse recording information from metadata into notes array
data['notes'] = []
obj = ('ClampState','ClampParams','DAQ','Protocol')
for i in range(4):
    if i==1:
        data['notes'].append(obj[i-1]+'.'+obj[i])
        d = metadata.get('2/'+obj[i-1]+'/'+obj[i]).attrs
    elif i==2:
        key = map(str,metadata.get('2/'+obj[i]).keys())
        data['notes'].append(obj[i])
        d = metadata.get('2/'+obj[i]).attrs
    else:
        data['notes'].append(obj[i])
        d = metadata.get('2/'+obj[i]).attrs
    key = map(str,d.keys())
    for j,keynow in enumerate(key):#range(1,len(key),1): 
        data['notes'].append('  %s: %s' %(keynow,str(d.get(keynow))))

# Calculate recording date and time and parse into 'saved' variable
from datetime import datetime, timedelta
recTime = [data['notes'][i][13:26]
          for i in range(len(data['notes']))
          if data['notes'][i][2:11] == 'startTime']
saved = datetime.fromtimestamp(float(recTime[0]))
from re import sub
data['saved'] = sub(r'\W+','',saved.isoformat()[0:-7])

   # return data