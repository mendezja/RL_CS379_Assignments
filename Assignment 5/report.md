# Assignment 5: Q-Learning Report
### Armando Mendez
.  
.  
.  

## Analysis

Solved using Q-learning for both slippery/not slippery.

I used a constant epsilon of .9 across all tests becuase the psudocode omited the use of greedy selection in exploration. Additionally, I implemented rand sample when all q values are equal to avoid bais. At smaller epsilon the number of iterations increased likely due to higher exploitation, which was detremantal given the stocastic nature of the environment.     

As expected, slippery=False was solved far quicker thand slipper=True, and as outlined in last assignment, there was little difference between seed and no-seed when looking at slippery=False.    
.  
.  

## Report: is_slippery: True | seed: None | Gamma: 0.9 | Espilon: 0.8 | Alpha: 0.2 

### Output

Best reward updated 0.000 -> 0.050  
Best reward updated 0.050 -> 0.100  
Best reward updated 0.100 -> 0.150  
Best reward updated 0.150 -> 0.200  
Best reward updated 0.200 -> 0.250  
Best reward updated 0.250 -> 0.350  
Best reward updated 0.350 -> 0.400   
Best reward updated 0.400 -> 0.450        
Best reward updated 0.450 -> 0.600  
Best reward updated 0.600 -> 0.650  
Best reward updated 0.650 -> 0.750  
Best reward updated 0.750 -> 0.800  
Best reward updated 0.800 -> 0.900  
*Solved in **6431** iterations!*  
.  
.  
.  

### Final Q Value Table:
  
| (s,a) | Left | Down | Right | Up |
|-------|-------|-------|--------|--------|
| 0 | 0.070 | 0.068 | 0.068 | 0.063 |
| 1 | 0.050 | 0.033 | 0.031 | 0.069 |
| 2 | 0.098 | 0.077 | 0.077 | 0.065 |
| 3 | 0.043 | 0.045 | 0.043 | 0.071 |
| 4 | 0.091 | 0.059 | 0.072 | 0.051 |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | 0.163 | 0.076 | 0.109 | 0.045 |
| 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | 0.063 | 0.081 | 0.122 | 0.137 |
| 9 | 0.100 | 0.219 | 0.082 | 0.126 |
| 10 | 0.234 | 0.220 | 0.232 | 0.130 |
| 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| 13 | 0.125 | 0.269 | 0.276 | 0.180 |
| 14 | 0.298 | 0.447 | 0.385 | 0.326 |
| 15 | 0.000 | 0.000 | 0.000 | 0.000 |

### Extracted Policy:

|       |       |       |       |
|-------|-------|-------|--------|
| LEFT  | UP    | LEFT  | UP    |
| LEFT  | LEFT  | LEFT  | LEFT  |
| UP    | DOWN  | LEFT  | RIGHT |
| RIGHT | RIGHT | DOWN  | RIGHT |



.  
.  
.  


## Report: is_slippery: True | seed: 42 | Gamma: 0.9 | Espilon: 0.8 | Alpha: 0.2 
.  
.  


### Output


Best reward updated 0.000 -> 0.050  
Best reward updated 0.050 -> 0.100  
Best reward updated 0.100 -> 0.150  
Best reward updated 0.150 -> 0.300  
Best reward updated 0.300 -> 0.350  
Best reward updated 0.350 -> 0.400  
Best reward updated 0.400 -> 1.000  
*Solved in **295** iterations!*  
.  
.  

### Q values:

| (s,a) | Left | Down | Right | Up |
|-------|-------|-------|--------|--------|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | 0.000 | 0.001 | 0.001 | 0.000 |
| 9 | 0.000 | 0.005 | 0.000 | 0.000 |
| 10 | 0.000 | 0.000 | 0.029 | 0.000 |
| 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| 13 | 0.000 | 0.000 | 0.000 | 0.000 |
| 14 | 0.000 | 0.200 | 0.000 | 0.000 |
| 15 | 0.000 | 0.000 | 0.000 | 0.000 |

### Policy:

|       |       |       |       |
|-------|-------|-------|--------|
| LEFT |  UP   |  LEFT |  UP   | 
| LEFT |  LEFT |  RIGHT|  LEFT | 
| UP   |  DOWN |  LEFT |  LEFT |
| LEFT |  RIGHT|  DOWN |  LEFT |

.  
.  
.

## Report: is_slippery: False | seed: None | Gamma: 0.9 | Espilon: 0.8 | Alpha: 0.2 
.  
.  
### Output


Best reward updated 0.000 -> 0.050  
Best reward updated 0.050 -> 0.100  
Best reward updated 0.100 -> 0.200  
Best reward updated 0.200 -> 0.250  
Best reward updated 0.250 -> 0.300  
Best reward updated 0.300 -> 0.400  
Best reward updated 0.400 -> 0.450  
Best reward updated 0.450 -> 0.550  
Best reward updated 0.550 -> 0.700  
Best reward updated 0.700 -> 0.800  
Best reward updated 0.800 -> 1.000  
*Solved in **302** iterations!*    
.   
.

### Q values:
| (s,a) | Left | Down | Right | Up |
|-------|-------|-------|--------|--------|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |   
| 1 | 0.000 | 0.000 | 0.000 | 0.000 |   
| 2 | 0.000 | 0.000 | 0.000 | 0.000 |   
| 3 | 0.000 | 0.000 | 0.000 | 0.000 |   
| 4 | 0.000 | 0.001 | 0.000 | 0.000 |   
| 5 | 0.000 | 0.000 | 0.000 | 0.000 |   
| 6 | 0.000 | 0.016 | 0.000 | 0.000 |   
| 7 | 0.000 | 0.000 | 0.000 | 0.000 |   
| 8 | 0.000 | 0.000 | 0.003 | 0.000 |   
| 9 | 0.001 | 0.000 | 0.016 | 0.000 |   
| 10 | 0.000 | 0.088 | 0.000 | 0.003 |  
| 11 | 0.000 | 0.000 | 0.000 | 0.000 |  
| 12 | 0.000 | 0.000 | 0.000 | 0.000 |  
| 13 | 0.000 | 0.000 | 0.000 | 0.000 |  
| 14 | 0.000 | 0.000 | 0.360 | 0.017 |  
| 15 | 0.000 | 0.000 | 0.000 | 0.000 |  

### Policy:

|       |       |       |       |
|-------|-------|-------|--------|
| DOWN  | DOWN  | RIGHT | LEFT  |
| DOWN  | RIGHT | DOWN  | RIGHT |
| RIGHT | RIGHT | DOWN  | DOWN  |
| LEFT  | LEFT  | RIGHT | DOWN  |