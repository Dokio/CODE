import collections
class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: the course order
    """
    def findOrder(self, numCourses, prerequisites):
        # write your code here
        if numCourses == 0:
            return []
        
        #building graph
        graph = [[] for _ in range(numCourses)]
        indegree = [0 for _ in range(numCourses)]
        
        for course_out, course_in in prerequisites: ##
            graph[course_in].append(course_out)
            indegree[course_out] += 1
            
        queue = collections.deque()
        result = []
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)
                result.append(i)
        
        while queue:
            course = queue.pop()
            for neighbor in graph[course]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
                    result.append(neighbor)
        
        if len(result) == numCourses:
            return result
        else:
            return []
                    
                
        