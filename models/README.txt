These are functional PRMS models that can be used for testing scripts.
Each model contains three input files: control, data, param 
PRMS executables are located in the /PRMS-Python/Dist directories for
windows and Linux compiled versions.  

Note, the file paths denoted for the control, param, data files, and prms app 
need to be set correctly in the control file. One option is to change them to 
have the name of the file only if they are all in the same working directory. 
This is quick, and is left to whatever you prefer, we will not be writing 
any code to adjust these settings. 

IMPORTANT: Text input files downloaded from USGS were for the windows version, if you want to use 
PRMS compiled for linux you will need to run the dos2unix command on them first else
PRMS will crash. Just sudo apt-get install dos2unix and then: 

$dos2unix filename

do this for the control, data, and param files if you plan on working on Linux. I have 
already done this for my model (lbcd) and it is ready to go for Linux. 

acf and merced models are both supplied examples from the USGS. Others are copies from
my masters thesis work but are not necessarily up to date calibrated versions. 
I removed most of the other files included with the USGS examples to avoid confusion-
this is really all we need. 

Let me know if you have any issues, I have not tested all the USGS examples but
ultimately we will want to test our code against at least these three models -John

