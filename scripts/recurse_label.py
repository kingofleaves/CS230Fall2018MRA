import glob

fileDir = '/data2/yeom/ky_mra'
files = glob.glob(fileDir + '/**/**/**/**/**/*.dcm')
print(files)
