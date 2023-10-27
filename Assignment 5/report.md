# Assignment 5: Q-Learning Report
### Armando Mendez
.  
.  
.  

## Analysis

Solved using value iteration for both slippery/not slippery, and seed/no seed.
.  
.  
.  

## Report: SLIPPERY=True | SEED=42

### Output

Best reward updated 0.000 -> 0.000  
Best reward updated 1.000 -> 1.000  
*Solved in **6** iterations!* .  
.  
.  

### Final State Values Table:
|       |       |       |       |
|-------|-------|-------|--------|
| 0.0733 | 0.1182 | 0.1990 | 0.1460 |
| 0.1652 | 0.0000 | 0.3547 | 0.0000 |
| 0.2779 | 0.6636 | 1.0071 | 0.0000 |
| 0.0000 | 1.0602 | 1.8325 | 0.0000 |

### Extracted Policy:

|       |       |       |       |
|-------|-------|-------|--------|
 DOWN  | DOWN  | LEFT  | UP    |
 LEFT  | LEFT  | RIGHT | LEFT  |
 DOWN  | DOWN  | LEFT  | LEFT  |
 LEFT  | RIGHT | RIGHT | LEFT  |



.  
.  
.  


## Report: SLIPPERY=True | SEED=None
.  
.  


### Output


Best reward updated 0.000 -> 0.000   
Best reward updated 0.050 -> 0.050   
Best reward updated 0.100 -> 0.100   
Best reward updated 0.200 -> 0.200     
Best reward updated 0.500 -> 0.500   
Best reward updated 0.550 -> 0.550   
Best reward updated 0.700 -> 0.700  
Best reward updated 0.750 -> 0.750  
Best reward updated 0.800 -> 0.800  
Best reward updated 0.850 -> 0.850  
*Solved in **33** iterations!*  
.  
.  

### State values:

|       |       |       |       |
|-------|-------|-------|--------|
| 0.2158 | 0.1983 | 0.2497 | 0.1885 | 
| 0.2950 | 0.0000 | 0.3792 | 0.0000 |
| 0.4937 | 0.8642 | 1.0820 | 0.0000 |
| 0.0000 | 1.4701 | 2.3927 | 0.0000 |

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

## Report: SLIPPERY=False | SEED=None
.  
.  
### Output


Best reward updated 0.000 -> 0.000   
Best reward updated 1.000 -> 1.000   
*Solved in **6** iterations!*    
.   
.

### State values:
|       |       |       |       |
|-------|-------|-------|--------|
 0.0000  0.6561  1.3365  1.2028 
 0.6561  0.0000  1.4850  0.0000 
 1.3365  1.4850  2.2050  0.0000 
 0.0000  2.2050  2.7179  0.0000 

### Policy:

|       |       |       |       |
|-------|-------|-------|--------|
 DOWN |   RIGHT |  DOWN |  LEFT  
 DOWN |  LEFT | DOWN   | LEFT  
 RIGHT|  DOWN  | DOWN  | LEFT  
 LEFT |  RIGHT | RIGHT | LEFT  