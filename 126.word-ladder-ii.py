#
# @lc app=leetcode id=126 lang=python3
#
# [126] Word Ladder II
#

# @lc code=start
'''
网上答案(未实现): 
BFS 搜索最短路径长度, 构造深度hashmap,根据hashmap和DFS反向搜索,
current dep+depth hashmap = min_depth
单纯BFS(ac):
把wordlist转换成set,与127.Wordladder中的BFS不同的是queue不是单纯的
存储路径,queue是一个dict[last_word] = [paths]
关键一步: 每层遍历完了以后,把Wordlist更新,把已经访问过的加入路径的单词删掉
这一步不仅实现去重,还有剪枝的功能其他路径不能在更深的时候访问该单词
'''
from collections import defaultdict
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        wordlist=set(wordList)
        return self.BFS(beginWord, endWord, wordlist)
    def BFS(self, beginWord, endWord, wordList):
        queue = defaultdict(list)
        queue[beginWord] = [[beginWord]]
        res = []
        while queue:
            new_queue = defaultdict(list)
            for w in queue:
                if w == endWord:
                    res+= queue[w]
                else:
                    for i in range(len(w)):
                        for c in "abcdefghijklmnopqrstuvwxyz":
                            new_word = w[:i] + c + w[i+1:]
                            if new_word in wordList:
                                new_queue[new_word] += [j+[new_word] for j in queue[w]]
            wordList -= set(new_queue.keys()) ## 关键一步:剪枝和去重
            queue = new_queue
        return res

    #     if endWord not in wordList: return []
    #     self.dic = defaultdict(list)
    #     if beginWord not in wordList: wordList.append(beginWord)
    #     for w in wordList:
    #         if len(w) != len(beginWord):
    #             continue
    #         for i in range(len(w)):
    #             temp = w[:i] + "*" + w[i+1:]
    #             self.dic[temp].append(w)
                
    #     return self.BFS_get_min_length(beginWord,endWord)

       
    
    # def BFS_get_min_length(self, beginWord, endWord):
    #     queue = [[beginWord]]
    #     res = []
    #     while queue:
    #         new_queue = []
    #         for i in range(len(queue)):
    #             current_path = queue[i]
    #             for i in range(len(current_path[-1])):
    #                 temp = current_path[-1][:i] + "*" + current_path[-1][i+1:]
    #                 for word in self.dic[temp]:
    #                     if word == endWord:
    #                         res.append(current_path+[word])
    #                     elif word not in current_path:
    #                         new_queue.append(current_path+[word])
    #         if res:
    #             return res
    #         queue = new_queue
    #     return res
    
    # def DFS(self,beginWord, endWord, min_depth):
    #     stack=[[endWord]]
    #     res = []
    #     while stack:
    #         current_path = stack.pop()
    #         current_deep = len(current_path)
    #         if current_deep > min_depth: continue
    #         elif current_deep == min_depth and current_path[-1] == beginWord:
    #             current_path.reverse()
    #             res.append(current_path)
    #         else:
    #             for i in range(len(current_path[-1])):
    #                 temp = current_path[-1][:i] + "*" + current_path[-1][i+1:]
    #                 for word in self.dic[temp]:
    #                     if word not in current_path and word in self.deep_dic \
    #                         and self.deep_dic[word] + current_deep - 1 < min_depth:
    #                         stack.append(current_path+[word])
            
    #     return res
# @lc code=end

