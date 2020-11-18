import collections
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """
    def numIslands(self, grid):
        # write your code here
        if not grid or not grid[0]:
            return 0
        
        n = len(grid)
        m = len(grid[0])
        visited = set()
        num_island = 0
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1 and (i, j) not in visited:
                    self.find_island_by_bfs(i, j, visited, grid)
                    num_island += 1
        return num_island
        
    
    def find_island_by_bfs(self, x, y, visited, grid):
        queue = collections.deque([(x,y)])
        visited.add((x,y))
        while queue:
            current_x, current_y = queue.popleft()
            for x_index, y_index in DIRECTIONS:
                next_x = current_x + x_index
                next_y = current_y + y_index
                if self.is_valid(next_x, next_y, visited, grid):
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))
        return
    
    def is_valid(self, x, y, visited, grid):
        if x >= len(grid) or y >= len(grid[0]) or x < 0 or y < 0:
            return False
        if (x, y) in visited:
            return False
        return grid[x][y]
        
        
                
                    
            
            
                    
                        