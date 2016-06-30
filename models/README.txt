These are functional PRMS models that can be used for PRMS-Python 
examples. Each model contains three input files: control, data, param 
PRMS executables are located in the /PRMS-Python/Dist directories for
windows and Linux compiled versions.  

Text input files downloaded from USGS were for windows systems, if you are using 
PRMS compiled for linux you need to run the dos2unix command on the text files first. 
Install dos2unix:

$sudo apt-get install dos2unix 

convert text formatting:

$dos2unix filename

run this command on the control, data, and param files if you plan on working on Linux. 
acf and merced models are both examples from the United States Geological Survey (USGS). 
