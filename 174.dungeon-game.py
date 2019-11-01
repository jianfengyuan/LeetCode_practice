#
# @lc app=leetcode id=174 lang=python3
#
# [174] Dungeon Game
#

# @lc code=start
'''
trick: 从公主房开始往起始点推
首先考虑每个位置的血量是由什么决定的，骑士会挂主要是因为去了下一个房间时，
掉血量大于本身的血值，而能去的房间只有右边和下边，所以当前位置的血量是由右
边和下边房间的可生存血量决定的，进一步来说，应该是由较小的可生存血量决定的，
因为较我们需要起始血量尽可能的少，因为我们是逆着往回推，骑士逆向进入房间后 
PK 后所剩的血量就是骑士正向进入房间时 pk 前的起始血量。所以用当前房间的右
边和下边房间中骑士的较小血量减去当前房间的数字，如果是负数或着0，说明当前房
间是正数，这样骑士进入当前房间后的生命值是1就行了，因为不会减血。而如果差是
正数的话，当前房间的血量可能是正数也可能是负数，但是骑士进入当前房间后的生
命值就一定要是这个差值。所以我们的状态转移方程是 
dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])。
'''
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        dp = [[0 for _ in range(len(dungeon[0]))] for _ in range(len(dungeon))]
        dp[-1][-1] =  max(1,1 - dungeon[-1][-1])
        for i in range(len(dp)-2,-1,-1):
            dp[i][-1] = max(1,max(1,dp[i+1][-1]) - dungeon[i][-1])
        for j in range(len(dp[0])-2,-1,-1):
            dp[-1][j] = max(max(1,dp[-1][j+1]) - dungeon[-1][j],1)
        for i in range(len(dp)-2,-1,-1):
            for j in range(len(dp[i])-2,-1,-1):
                dp[i][j] = max(1,min(dp[i+1][j],dp[i][j+1])- dungeon[i][j])
        return dp[0][0]
# @lc code=end

