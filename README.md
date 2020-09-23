Dear Reader,

This is a repository for simulation codes and input/output files used in the hypertension treatment coverage and adherence study. In addition to the files provided here, we used the WHO Sage Wave 1 dataset and the Globorisk calculator, both which were received upon request from the respective authors, and have not been shared in the online repository.
The simulation codes are not commit-ready and a user will have to change the directories to store the output files in either their local drive or Dropbox account. 

There are broadly five steps to reproduce the results:
1. Create a cleaned dataset from the WHO SAGE Wave 1 dataset.
2. Run the Cohort Creation.R code to create a hypothetical cohort of 10000 females and 10000 males, with the age-related increase in systolic blood pressure incorporated.
3. The 10 year CVD risk is calulated using the Globorisk calculator STATA code (not provided in the repository) to create the input file for the simulation.
4. The cohort (within Python Cohort Import) is simulated with Public Sector Simulate Code.py to create a csv wherein each row contains the output of each simulation run.
5. The resulting simulation output files (within Simulation Outputs) can be analysed to calculate the ICER and other outcomes of interest using the Summary Results.py. The output from the code are available in the webappendix along with the paper.

To perform sensitivity analysis, the user will have to change the relevant parameters in line# 735, 765, 766 of the simulation code (step# 4). 

Also, the run time of each scenario with 1000 simulation takes approximately 6 days if run on single core processor (3.3Ghz). We leveraged AWS EC2 servers and Python multiprocessing ability to run the simulation runs of each scenario in parallel to generate the simulation outputs faster. The same has been implemented in the python codes, but the user will have to assess the number of available cores and modify the scenarios that are run in parallel.

Thanks. 
