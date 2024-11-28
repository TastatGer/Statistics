# Statistics

#Description
this programm calculates Kaplan-Meier Curves or cummulative incidence for a set of Patient Survival Data. 
The Data has to be in an Excel.
I sorted the Patient Data in 9 columns as followed:
          ID    OStime  LRCtime  OS    LRC    Age    GTV   TBR    HV
          1     1       1        1     0      50     30    1,2    0
dtype    uint   uint    uint    [0;1]  [0;1]  uint   uint  float  uint
ID                 - Patient ID
OStime             - time to last information
LRCtime            - time to relapse
OS                 - 0 = censor, 1 = death
LRC                - 0 = no relapse, 1 = relapse
Age, GTV, TBR, HV  -  arbitrary risk factors

i can recommend Applied Survival Analysis from Hosmer, Lemeshow to read up on cumulative incidence.
its a good source to double check or look up formulas.

#Dependencies
  python 3.9.13
  pandas 1.4.4
  numpy 1.21.5
  scipy 1.9.1
  matplotlib 3.5.2

#For questions, please open an issue or start a discussion in this repository or contact me on discord tastat#3604.
