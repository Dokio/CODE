"""
Definition for a Directed graph node
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""

import collections
class Solution:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # write your code here
        if not graph:
            return []
        
        node_to_indegree = self.find_indegree(graph)
        # queue = collections.deque()
        # for x in node_to_indegree:
        #     if node_to_indegree[x] == 0:
        #         queue.append(x)  ###
        queue = collections.deque(x for x in graph if node_to_indegree[x] == 0)
        result = [x for x in graph if node_to_indegree[x] == 0]
        while queue:
            node = queue.pop()
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] -= 1
                if node_to_indegree[neighbor] == 0:
                    queue.append(neighbor)
                    result.append(neighbor)
                # for x in node_to_indegree:
                #     if node_to_indegree[x] == 0:
                #     queue.append(x)
                #     result.append(x)
        return result
            
        
        
        
    def find_indegree(self, graph):
        node_to_indegree = {x : 0 for x in graph}
        for node in graph:
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] += 1
        return node_to_indegree
        
                
                
        
            
        