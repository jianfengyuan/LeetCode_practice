class Solution:
    def FirstNotRepeatingChar(self, s):
            # write code here
            d = {}
            if len(s) == 0:
                return -1
            for i in range(len(s)):
                if s[i] in d:
                    d[s[i]][1]+=1
                else:
                    d[s[i]] = [i,1]
            res = sorted((filter(lambda x: x[1][1]==1,s.FirstNotRepeatingChar("google").items())),key=lambda x:x[1][0])
            if res:
                return res[0][1][0]
            else:
                return -1

s = Solution()

print(sorted((filter(lambda x: x[1][1]==1,s.FirstNotRepeatingChar("googllee").items())),key=lambda x:x[1][0]))

# print(s.FirstNotRepeatingChar("google").items())