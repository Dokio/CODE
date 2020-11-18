import heapq
class Solution:
    """
    @param words: a list of words
    @return: a string which is correct order
    """
    def alienOrder(self, words):
        # Write your code here
        graph = self.get_graph(words)
        indegree = self.get_indegree(graph)
        return self.get_result(graph, indegree)
        
    
    def get_graph(self, words):
        graph = {}
        for word in words:
            for node in word:
                if node not in graph:
                    graph[node] = set()
        
        for i in range(1, len(words)):
            for j in range(min(len(words[i- 1]), len(words[i]))):
                if words[i-1][j] != words[i][j]:
                    graph[words[i-1][j]].add(words[i][j])
                    break
                if j == min(len(words[i- 1]), len(words[i])) :
                    if len(words[i- 1]) > len(words[i]):
                        return ''
        return graph
        
    def get_indegree(self, graph):
        indegree = {node : 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegree[neighbor] += 1
        return indegree
        
    def get_result(self, graph, indegree):
        queue = [node for node in indegree if indegree[node] == 0]
        heapq.heapify(queue)
        result = ''
        while queue:
            node = heapq.heappop(queue)
            #result = ''.join(result, node)
            result += node
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    heapq.heappush(queue, neighbor)
                    
        if len(result) == len(graph):
            return result
        
        return ''
            
        
            
    
    
        
    
                
        
            
                
             
                
            
            
        
        
        
        
    
    