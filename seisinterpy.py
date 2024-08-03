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
