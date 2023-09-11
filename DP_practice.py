# Given n non-negative integers representing an elevation map where the width of each bar is 1, compute
# how much water it can trap after raining.

# Example:

# Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6
# The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6
# units of rainwater (blue section) are being trapped

def waterTrapped(heights):
    size = len(heights)
    
    left_max = [-1] * size
    left_max[-1] = heights[-1]

    right_max = [-1] * size
    right_max[0] = heights[0]
    
    # print(right_max)
    # print(left_max)
    
    for x in range(1,size):
        right_max[x] = max(right_max[x-1], heights[x])

        l_x = (size-1) - x
        left_max[l_x] = max(left_max[l_x +1], heights[l_x])
    print(heights)
    print(right_max)
    print(left_max)
        # print()
    water_sum = 0
    for x in range(size):
        water_sum += min(right_max[x]-heights[x], left_max[x]-heights[x])
    return water_sum


height = [0,1,0,2,1,0,1,3,2,1,2,1]
ans = [0,0,1,0,1,2,1,0,0,1,0,0]
print(waterTrapped(height))
print(ans)