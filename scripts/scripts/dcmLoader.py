import os
import sys
import numpy as np
import pydicom


# filepathExample = "/data2/yeom/ky_mra/Normal_MRA/mri_anon/N001/StanfordNormal/MRA_3DTOF_MT_-_4/IM-0001-0001-0001.dcm"

class DCMLoader:
  def __init__(self):

  def loadData(filepaths):
    read_data = []
    if len(sys.argv) == 1:
      print("argument requires filepath(s) to be loaded")
      sys.exit()

    filepaths = sys.argv[1:]
    for input in filepaths:
      print("Processing %s" % input)
      data = pydicom.dcmread(input)
      data_np = data.pixel_array
      read_data.append(data_np)
    
    return read_data


