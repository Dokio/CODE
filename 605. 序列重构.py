import collections
class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        # write your code here
        graph = self.get_graph(seqs)
        indegree = self.get_indegree(graph)
        return self.get_org_by_bfs(graph, indegree, org)
        
       
    def get_graph(self, seqs):
        graph = {}
        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = set()
                    
        for seq in seqs:
            for i in range(1, len(seq)):
                graph[seq[i - 1]].add(seq[i])
        return graph
        
    def get_indegree(self, graph):
        indegree = {node : 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegree[neighbor] += 1
        return indegree
        
    def get_org_by_bfs(self, graph, indegree, org):
        queue = collections.deque(node for node in indegree if indegree[node] == 0)
        if len(queue) > 1:
            return False
        result = [node for node in indegree if indegree[node] == 0]
        while queue:
            node = queue.pop()
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
                    if len(queue) > 1:
                        return False
                    result.append(neighbor)
        if result == org:
            return True
        else:
            return False

            
    
    
    
        
                
    
                
        
        
                
        
        
