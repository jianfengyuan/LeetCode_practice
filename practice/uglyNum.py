uglynum = [1]
n = int(input())
i = 1
t2 = m2 = 0
t3 = m3 = 0
t5 = m5 = 0
while i < n:
    for x in range(t2, len(uglynum)):
        m2 = uglynum[x]*2
        # print("m2:",m2,end=" ")
        if m2 > uglynum[-1]:
            t2 = x
            # print("t2:",t2)
            break	
    for x in range(t3, len(uglynum)):
        m3 = uglynum[x]*3
        # print("m3:",m3,end=" ")
        if m3 > uglynum[-1]:
            t3 = x
            # print("t3:",t3)
            break
    for x in range(t5, len(uglynum)):
        m5 = uglynum[x]*5
        # print("m5",m5,end=" ")
        if m5 > uglynum[-1]:
            t5 = x
            # print("t5:",t5)
            break
    uglynum.append(min(m2,m3,m5))
    i += 1
print(uglynum)
'''
 试图只计算丑数，而不在非丑数的整数上花费时间。根据丑数的定义，丑数应该是另一个丑数乘以2、3或者5的结果（1除外）。
 因此我们可以创建一个数组，里面的数字是排好序的丑数。里面的每一个丑数是前面的丑数乘以2、3或者5得到的。
 这种思路的关键在于怎样确保数组里面的丑数是排好序的。我们假设数组中已经有若干个丑数，排好序后存在数组中。
 我们把现有的最大丑数记做M。现在我们来生成下一个丑数，该丑数肯定是前面某一个丑数乘以2、3或者5的结果。我们首先考虑把已有的
 每个丑数乘以2。在乘以2的时候，能得到若干个结果小于或等于M的。由于我们是按照顺序生成的，小于或者等于M肯定已经在数组中了，
 我们不需再次考虑；我们还会得到若干个大于M的结果，但我们只需要第一个大于M的结果，因为我们希望丑数是按从小到大顺序生成的，
 其他更大的结果我们以后再说。我们把得到的第一个乘以2后大于M的结果，记为M2。同样我们把已有的每一个丑数乘以3和5，能得到第一
 个大于M的结果M3和M5。那么下一个丑数应该是M2、M3和M5三个数的最小者。
 前面我们分析的时候，提到把已有的每个丑数分别都乘以2、3和5，事实上是不需要的，因为已有的丑数是按顺序存在数组中的。
对乘以2而言，肯定存在某一个丑数T2，排在它之前的每一个丑数乘以2得到的结果都会小于已有最大的丑数，在它之后的每一个丑数乘
以2得到的结果都会太大。我们只需要记下这个丑数的位置，同时每次生成新的丑数的时候，去更新这个T2。对乘以3和5而言，存在着同样
的T3和T5。
'''