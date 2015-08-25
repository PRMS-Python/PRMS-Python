# -*- coding: utf-8 -*-
"""
Quickly modify parameters from any PRMS parameter file by a multiplier
and have option to run PRMS
John Volk 02/25/2015 
Python 2
"""
import os
import numpy as np
import subprocess

#### workspace
workspace = os.getcwd()

# file names, don't forget to change these for your model
paramfilename = 'lbcd.param'

controlfilename = 'lbcd_hist.control'

programname = 'prms.exe'

#### create file paths
paramfile = os.path.join(workspace,paramfilename) 
controlfile = os.path.join(workspace,controlfilename)
program = os.path.join(workspace,programname)

# <codecell>

def loadfile(filepath=paramfile):
    with open(filepath, 'r') as paramdata:
        paramlines = paramdata.readlines()
    return paramlines

# <codecell>

def getparam(lines,paramName,multiplier):
    starti = -1
    for i,line in enumerate(lines):
        if paramName == line.split()[0]:
            starti = i
            try:
                dims = str(lines[starti +1].strip())
            except:
                print 'There was an error reading the dimension info for {}'.format(paramName)
                break
                
            if dims == '1':
                try:
                    paramDim = str(lines[starti +2].strip()) 
                    ndims = int(lines[starti +3].strip())
                    datatype = int(lines[starti+4].strip())
                except:
                    print 'There was an error reading the dimension info for {}'.format(paramName)
                    break                        
                print 'Parameter: ' ,paramName
                print 'Dimension: ' , paramDim
                print 'Number of dimensions: ' , ndims
                if paramDim != "nhru" and paramDim != 'nmonths' and paramDim != 'one':
                    print 'Sorry {} is not a parameter I was programmed to change!'.format(paramName)
                    break
                datastart = starti + 5
                dataend = datastart + ndims
                paramdata = lines[datastart:dataend]
                paramdata = np.array(paramdata)
                if str(datatype) == '1':
                    paramdata = paramdata.astype(int)
                    ans = raw_input('Are you sure you want to multiply {0} which is of type int by {1:.2f}? y/n: '.format(paramName,multiplier))
                    if ans == 'y' or ans == 'Y':
                        newparam = paramdata * multiplier
                        newparam = newparam.astype(int)
                    else:
                        break
                elif str(datatype) == '2':
                    paramdata = paramdata.astype(float)
                    newparam = paramdata * multiplier
                elif str(datatype) == '4':
                    print "Sorry I can't modify {} which is a string data type in PRMS".format(paramName)
                ## don't continue to loop through file after the param was found
                print 'Success, all {0} values were multiplied by {1:.5f}\n'.format(paramName,multiplier)
                return {'newparam' :newparam , 'startline':datastart , 'endline': dataend} ## will only return if param found
                break
                
            ##In case we are dealing with two dimensional parameters (snow rain adjust, etc)    
            elif dims == '2':
                try:
                    paramDim1 = str(lines[starti +2].strip()) 
                    paramDim2 = str(lines[starti +3].strip()) 
                    ndims = int(lines[starti +4].strip())
                    datatype = int(lines[starti+5].strip())
                except:
                    print 'There was an error reading the dimension info for {}'.format(paramName)
                    break
                print 'Parameter: ' ,paramName
                print 'Dimensions: {} by {}'.format(paramDim1, paramDim2)
                print 'Number of dimensions: ' , ndims
                datastart = starti + 6
                dataend = datastart + ndims
                paramdata = lines[datastart:dataend]
                paramdata = np.array(paramdata)
                if str(datatype) == '1':
                    paramdata = paramdata.astype(int)
                    ans = raw_input('Are you sure you want to multiply {0} which is of type int by {1:.5f}? y/n: '.format(paramName,multiplier))
                    if ans == 'y' or ans == 'Y':
                        newparam = paramdata * multiplier
                        newparam = newparam.astype(int)
                    else:
                        break
                elif str(datatype) == '2':
                    paramdata = paramdata.astype(float)
                    newparam = paramdata * multiplier
                elif str(datatype) == '4':
                    print "Sorry I can't modify {} which is a string data type in PRMS".format(paramName)
                ## don't continue to loop through file after the param was found
                print 'Success, all {0} values were multiplied by {1:.5f}\n'.format(paramName,multiplier)
                return {'newparam' :newparam , 'startline':datastart , 'endline': dataend} ## will only return if param found
                break
    if starti == -1:
        print 'The parameter {} was not found in the file'.format(paramName)

# <codecell>

def modifyFile(result,paramlines,outfile=paramfile):
    ##modify existing file with new param
    start = result['startline']
    end = result['endline']
    data = result['newparam']
    data = [str(x) + '\n' for x in data]
    paramlines[start:end] = data
    ## rewrite entire file
    with open(outfile, 'w+') as paramout:
        paramout.writelines(paramlines)

# <codecell>

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

# <codecell>

def run():
    done = False
    while not done:
        
        ####try to load parameter file
        try:
            datalines = loadfile(paramfile)
        except:
            print 'Error loading parameter file: make sure it exists in the current directory and\nthe file name is correct in changeParam.py'
            break
        #### Get parameter we want to modify from user
        paramName = str(raw_input('Enter a parameter you want to change, r to run prms or q to quit: '))
        if paramName.lower() == 'q':
            done = True
            break
        if paramName.lower() == 'r':
            try:
                subprocess.call([program,controlfile])
                continue
            except:
                print 'Error running PRMS, make sure the correct executable, data and control file are\n\
                        in the directory and file names are correct in changeParam.py'
        elif is_number(paramName):
            print "Invalid parameter name, you entered a number"
            continue
        ## get a numeric multiplier to multiply our parameter
        multiplier = raw_input('Enter a multiplier: ')
        try:
            multiplier = float(multiplier)
        except:
            print 'You did not enter a valid numeric multiplier, it needs to be an integer or a float'
            continue
        ## extract and modify parameter if possible
        result = getparam(datalines,paramName,multiplier)
        if result is None:
            continue
        ## rewrite changes to file in place
        modifyFile(result,datalines,paramfile)

# <codecell>
print '\n-----------------------------------------------------------------'
run()    

