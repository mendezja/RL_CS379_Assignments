# Given n non-negative integers representing an elevation map where the width of each bar is 1, compute
# how much water it can trap after raining.

# Example:

# Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6
# The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6
# units of rainwater (blue section) are being trapped

def waterTrapped(heights):
    size = len(heights)
    
    left_max = list(heights)
    right_max = list(heights)

    
    for i in range(1,size):
        left_max[i] = max(left_max[i-1], left_max[i])

        r_i = (size-1) - i

        right_max[r_i] = max(right_max[r_i +1], right_max[r_i])
    
    return sum(map(lambda x,y,z: min(x-z, y-z), right_max, left_max, height))



height = [0,1,0,2,1,0,1,3,2,1,2,1]
ans = [0,0,1,0,1,2,1,0,0,1,0,0]
print(waterTrapped(height))
# print(ans)