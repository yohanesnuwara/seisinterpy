import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import interpn
import segyio

def openSegy3D(filename, iline=189, xline=193):
  """
  Open 3D seismic volume in SEGY or SGY format

  NOTE: Default byte location iline=189, xline=193
        If it returns "openSegy3D cannot read the data", change iline and xline
        Usually, specifying iline=5, xline=21 works
  """
  import segyio

  try:
    with segyio.open(filename) as f:

      data = segyio.tools.cube(f)

      inlines = f.ilines
      crosslines = f.xlines
      twt = f.samples
      sample_rate = segyio.tools.dt(f) / 1000

      print('Successfully read \n')
      print('Inline range from', inlines[0], 'to', inlines[-1])
      print('Crossline range from', crosslines[0], 'to', crosslines[-1])
      print('TWT from', twt[0], 'to', twt[-1])
      print('Sample rate:', sample_rate, 'ms')

      try:
        rot, cdpx, cdpy = segyio.tools.rotation(f, line="fast")
        print('Survey rotation: {:.2f} deg'.format(rot))
      except:
        print("Survey rotation not recognized")

      cube = dict({"data": data,
                   "inlines": inlines,
                   "crosslines": crosslines,
                   "twt": twt,
                   "sample_rate": sample_rate})

      # See Stackoverflow: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
      class AttrDict(dict):
          def __init__(self, *args, **kwargs):
              super(AttrDict, self).__init__(*args, **kwargs)
              self.__dict__ = self

      cube = AttrDict(cube)

  except:
    try:
      # Specify iline and xline to segyio where to read
      with segyio.open(filename, iline=iline, xline=xline) as f:

        data = segyio.tools.cube(f)

        inlines = f.ilines
        crosslines = f.xlines
        twt = f.samples
        sample_rate = segyio.tools.dt(f) / 1000

        print('Successfully read \n')
        print('Inline range from', inlines[0], 'to', inlines[-1])
        print('Crossline range from', crosslines[0], 'to', crosslines[-1])
        print('TWT from', twt[0], 'to', twt[-1])
        print('Sample rate:', sample_rate, 'ms')

        try:
          rot, cdpx, cdpy = segyio.tools.rotation(f, line="fast")
          print('Survey rotation: {:.2f} deg'.format(rot))
        except:
          print("Survey rotation not recognized")

        cube = dict({"data": data,
                     "inlines": inlines,
                     "crosslines": crosslines,
                     "twt": twt,
                     "sample_rate": sample_rate})

        # See Stackoverflow: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self

        cube = AttrDict(cube)

    except:
      print("openSegy3D cannot read the data")
      cube = None

  return cube

def sliceViewer(cube, cube_name=" "):
  """
  Interactive viewer of 2D slice from a 3D seismic volume data

  NOTE: Interactive display. If you want static display, use: sliceCube

  """
  from ipywidgets import interact, interactive, fixed, interact_manual, ToggleButtons
  import ipywidgets as widgets
  import numpy as np
  import matplotlib.pyplot as plt

  # Access data and properties of cube
  data, inlines, crosslines, twt, sample_rate = cube.data, cube.inlines,\
                                                cube.crosslines, cube.twt,\
                                                cube.sample_rate                                         

  # Interactive plotting
  type = ToggleButtons(description='Selection',
                       options=['Inline','Crossline','Timeslice'])
  cmap_button = ToggleButtons(description='Colormaps',
                              options=['gray','seismic','RdBu','PuOr','Accent'])
  
  inline_loc = widgets.IntSlider(value=min(inlines), min=min(inlines), 
                                 max=max(inlines))
  xline_loc = widgets.IntSlider(value=min(crosslines), min=min(crosslines), 
                                max=max(crosslines))
  timeslice_loc = widgets.FloatSlider(value=min(twt), min=min(twt), 
                                      max=max(twt), step=sample_rate)
  vmin = widgets.FloatSlider(value=np.amin(data), min=np.amin(data), 
                             max=np.amax(data))
  vmax = widgets.FloatSlider(value=np.amax(data), min=np.amin(data), 
                             max=np.amax(data))

  @interact   
  def f(type=type, inline_loc=inline_loc, xline_loc=xline_loc,
        timeslice_loc=timeslice_loc, vmin=vmin, vmax=vmax, cmap=cmap_button):  
    
    if type == 'Inline':

      # with segyio.open(filename) as f: 
      inline_slice = data[inline_loc-inlines[0],:,:]  

      plt.figure(figsize=(20, 10))
      plt.title('{} Seismic at Inline {}'.format(cube_name, inline_loc))
      extent = [crosslines[0], crosslines[-1], twt[-1], twt[0]]

      p1 = plt.imshow(inline_slice.T, cmap=cmap, aspect='auto', extent=extent,
                      vmin=vmin, vmax=vmax, interpolation='bicubic')
      plt.colorbar()

      plt.xlabel('Crossline'); plt.ylabel('TWT')
      plt.show()

    if type == 'Crossline':

      # with segyio.open(filename) as f:
      xline_slice = data[:,xline_loc-crosslines[0],:]  

      plt.figure(figsize=(20, 10))
      plt.title('{} Seismic at Crossline {}'.format(cube_name, xline_loc))
      extent = [inlines[0], inlines[-1], twt[-1], twt[0]]

      p1 = plt.imshow(xline_slice.T, cmap=cmap, aspect='auto', extent=extent, 
                      vmin=vmin, vmax=vmax, interpolation='bicubic')
      plt.colorbar()

      plt.xlabel('Inline'); plt.ylabel('TWT')
      plt.show()
    
    if type == 'Timeslice':

      id = np.where(twt == timeslice_loc)[0][0]
      tslice = data[:,:,id]

      plt.figure(figsize=(8,10))
      plt.title('{} Seismic at Timeslice {} ms'.format(cube_name, timeslice_loc))
      extent = [inlines[0], inlines[-1], crosslines[-1], crosslines[0]]

      p1 = plt.imshow(tslice.T, cmap=cmap, aspect='auto', extent=extent, 
                      vmin=vmin, vmax=vmax, interpolation='bicubic')
      plt.colorbar()


      plt.xlabel('Inline'); plt.ylabel('Crossline')
      plt.gca().xaxis.set_ticks_position('top') # axis on top
      plt.gca().xaxis.set_label_position('top') # label on top
      plt.xlim(min(inlines), max(inlines))
      plt.ylim(min(crosslines), max(crosslines))
      plt.axis('equal')
      plt.show()   

def MeanSqAmplitude(x, w):
  return np.convolve(np.square(x), np.ones(w)/w, 'same')

def RMSAmplitude(x, w):
  return np.sqrt(MeanSqAmplitude(x, w))

def RMSAmplitudeCube(cube):
  data = cube.data

  arms = []
  for i in range(data.shape[0]):
      for j in range(data.shape[1]):
          # a = np.square(data[i,j])
          # mask = np.ones(11)/11
          # RMS_amp = np.sqrt(np.convolve(a, mask, 'same'))
          RMS_amp = RMSAmplitude(data[i,j], 20)
          arms.append(RMS_amp)
  ARMS = np.array(arms)
  ARMS = np.reshape(ARMS, data.shape)
  return ARMS  

def extractCDP(filename):
  # Get attributes and their byte locations
  header = segyio.tracefield.keys

  # Get byte location of an attribute
  cdpx_byte, cdpy_byte = header['SourceX'], header['SourceY']

  CDPX, CDPY = [], []
  with segyio.open(filename, ignore_geometry=True) as f:
    # Calculate number of trace
    n_traces = f.tracecount

    # List of CDPs per trace
    cdpx = [(f.attributes(cdpx_byte)[i])[0] for i in range(n_traces)]
    cdpy = [(f.attributes(cdpy_byte)[i])[0] for i in range(n_traces)]
    CDPX.append(cdpx)
    CDPY.append(cdpy)  
  return CDPX, CDPY

def seismicHorizon(filename, colnames=["A", "B", "UTM X", "UTM Y", "TWT"]):
  horizon = np.loadtxt(filename)
  horizon = pd.DataFrame(horizon, columns=colnames)
  return horizon 

def extractAttributeOnHorizon(CDPX, CDPY, cube_seismic, cube_attribute, horizon):
  """
  Extract attribute on seismic horizon
  """
  # Define the min and max of value range (seismic cube)
  min_xi, max_xi = np.amin(CDPX), np.amax(CDPX) # Is the min and max of coordinates of the cube data
  min_yi, max_yi = np.amin(CDPY), np.amax(CDPY)
    
  # Define dimension of seismic cube
  nx, ny, nz = cube_seismic.data.shape

  # Create array of coordinates
  xi = np.linspace(min_xi, max_xi, nx)
  yi = np.linspace(min_yi, max_yi, ny)
  zi = cube.twt

  # Make into tuple
  X_train = (xi, yi, zi) # Tuple

  y_train = cube_attribute # 3D numpy array

  X_test = horizon[['UTM X', 'UTM Y', 'TWT']].values

  y_pred = interpn(X_train, y_train, X_test, bounds_error=False)

  return y_pred
