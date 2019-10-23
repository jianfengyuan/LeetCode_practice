#
# @lc app=leetcode id=127 lang=python3
#
# [127] Word Ladder
#

# @lc code=start
import collections
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not beginWord or not endWord or not wordList \
            or endWord not in wordList :
            return 0 
        L = len(beginWord)
        word_dict = collections.defaultdict(list)
        for word in wordList:
            if len(word)!= L:
                continue
            for l in range(L):
                temp = word[:l] + "*" + word[l+1:]
                word_dict[temp].append(word)
        visited = {beginWord}
        length = 1
        queue = [beginWord]
        while queue:
            for _ in range(len(queue)):
                word = queue.pop(0)
                for i in range(len(word)):
                    temp = word[:i] + "*" + word[i+1:]
                    for w in word_dict[temp]:
                        if w == endWord:
                            return length + 1
                        if w not in visited:
                            queue.append(w)
                            visited.add(w)
                        
            length += 1
        return 0


# @lc code=end

