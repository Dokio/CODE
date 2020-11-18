import collections
class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def canFinish(self, numCourses, prerequisites):
        # write your code here
        if numCourses == 0 or numCourses == 1:
            return True
        if not prerequisites:
            return True
        
        
        graph = self.get_graph(numCourses, prerequisites)
        indegree = self.get_indegree(graph)
        result = self.get_result_by_topo(graph, indegree)
        if len(result) == numCourses:
            return True
        return False
        
    def get_graph(self, numCourses, prerequisites):
        graph = {i : [] for i in range(numCourses)}
        for course_next, course_first in prerequisites:
            graph[course_first].append(course_next)
        return graph
            
    def get_indegree(self, graph):
        indegree = {node : 0 for node in graph}
        #print (indegree)
        for node in graph:
            for neighbor in graph[node]:
                indegree[neighbor] += 1
        return indegree
        
    def get_result_by_topo(self, graph, indegree):
        queue = collections.deque([node for node in indegree if indegree[node] == 0])
        result = [node for node in indegree if indegree[node] == 0]
        while queue:
            node = queue.pop()
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor]  == 0:
                    queue.append(neighbor)
                    result.append(neighbor)
        return result
                
        
        
    
            
    
        
            
