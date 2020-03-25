# Auto-ASD-Network

## Research article 
Taban Eslami, Fahad Saeed (2019) **Auto-ASD-Network: A technique based on Deep Learning and Support Vector Machines for diagnosing Autism Spectrum Disorder using fMRI data**, Proceedings of the 10th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics, Niagara Falls, NY

## Enviroment Setup:
The Specification of system and libraries which used for testing the code are as below:

- Linux operating system
- atm (Auto Tune Models) Version: 0.2.2.dev0 (Home-page: https://github.com/HDI-project/ATM) 
- Pytorch
- scikitlearn
- CUDA version 8 or above

## Data:
  To download the fMRI data used for training the model run the following command:
  
  `python download.py`

## To run the code:
  `bash script.sh 'site_name'`
  
  In which site_name variable can be one of the followings:
  [USM,NYU,UCLA,OHSU], e.g.
 
 `bash script.sh 'UCLA'`

The result of each algorithm will be stored in a text file inside a folder named results. 

In case of any questions please contact: taban.eslami@wmich.edu
