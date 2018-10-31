import glob

fileDir = '/data2/yeom/ky_mra'
pattern_match = '/**/**/**/**/*.dcm'

# End up with two arrays of all normal file names and all abnormal file names
normals = glob.glob(fileDir + '/Normal_MRA' + pattern_match)
abnormals = glob.glob(fileDir + '/MMD_MRA' + pattern_match)

