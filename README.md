# svrLSMpy
PYthon code for Lesion Symptom Mapping using Support Vector Regression

## Getting started
in the symptoms folder,  
make a folder with the symptom name, (for example: example_symptom)  
in example_symptom, place your behavioral scores csv file (2 columns, 'filename': containing the full filenames of the binary lesion files, including the file extension i.e. .nii or .nii.gz; and 'behavior' which contains the corresponding behavioral scores of the subject).  
In the example_symptom folder make a folder called data and place the binary lesion files into it.

and run main.py  

the output of a single SVR LSM iteration is contained in a folder which is dynamically named according to the symptom name, the number of permutations and the timestamp when the run starts

