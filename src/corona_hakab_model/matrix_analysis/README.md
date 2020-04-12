# **Matrix Analyze Module**  
  
This module calculates histograms and statistics about the connections represented in the matrix.  
  
The module runs from the main API using 'analyze-matrix' command.  
  
  ## **INPUTS**:  
  **--matrix** <path to matrix> (optional) : This input specifies the path to the matrix which will be analyzed. If not given, assuming the 
  matrix file is at **output** folder.  
  **--show** (optional) : This input determines whether or not histogram plots will be shown (they are always being saved).  
  
  ****NOTICE**: When plots are being shown, you need to close them in order for the program to finish.  
    
  ## **OUTPUTS**  
  The module creates multiple files. The files created are raw-data .csv file of the matrix's different connection type (e.g. work, family) 
  and an histogram analysis of those connections (both .csv file and .png of the histogrm).  
  The files are being saved to **'../../output/matrix_analysis/'** folder.  
  The created folder is divided to 2 subfolders:  
  **histogram_analysis** - This folder is where the histogram plots and .csv files are saved.  
  **raw_matrices** - This folder is where the matrix's raw data is saved.  
  
  ## **EXAMPLES**  
  1. Run the analysis on defualt matrix file and don't show plots:  
  `./main analyze-matrix`  
  2. Run with specified matrix file:  
  `./main analyze-matrix --matrix ../../example/to/matrix/path`  
  3. Run with specified matrix file and show histograms:  
  `./main analyze-matrix --matrix ../../example/to/matrix/path --show`  
  
