"""
Definition for a undirected graph node
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""

import collections
class Solution:
    """
    @param {UndirectedGraphNode[]} nodes a array of undirected graph node
    @return {int[][]} a connected set of a undirected graph
    """
    def connectedSet(self, nodes):
        # write your code here
        if not nodes:
            return [[]]
            
        results = []
        visited = set()
        for node in nodes:
            if node not in visited:
                result = self.find_result_by_bfs(node, visited)
                results.append(sorted(result))
        return results
        
    def find_result_by_bfs(self, node, visited):
        queue = collections.deque([node])
        visited.add(node)
        result = [node.label]
        while queue:
            node = queue.popleft()
            for neighbor in node.neighbors:
                if neighbor in visited:
                    continue
                queue.append(neighbor)
                visited.add(neighbor)
                result.append(neighbor.label)
        return result
                    
        
        
            
            
