# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:52:55 2025

@author: pinto
"""

   def spawn_resources(self, randomize):        
       #np.random.seed(int(time.time()))
       
       while True:
           
           self.storage_layer = np.zeros(self.field_size, np.int32)
           self.network_layer = np.zeros(self.field_size, np.int32)
           
           total_required_resources = self.min_consumption * len(self.players)  # Total consumption goal
           total_spawned_resources = 0  # Track total resource levels spawned
       
           storage_count = 0
           network_count = 0
           attempts = 0
       
           while (total_spawned_resources < total_required_resources or
                  storage_count < self.num_storage or
                  network_count < self.num_network) and attempts < 2000:
               
               attempts += 1
               
               row, col = np.random.randint(1, self.rows-1), np.random.randint(1, self.cols-1)
               
               # Check if it has neighbors:
               
               if (
                   self.neighborhood(row, col).sum() == 0
                   and self.neighborhood(row, col, distance=2, ignore_diag=True) == 0
                   and self._is_empty_location(row, col)
               ):
                   
                   if storage_count < self.num_storage:
                       # Random level between 1 and storage_level
                       level = np.random.randint(1, self.storage_level + 1) if randomize else self.storage_level
                       self.storage_layer[row, col] = level
                       total_spawned_resources += level
                       storage_count += 1
                       continue
               
                   if network_count < self.num_network:
                      # Random level between 1 and network_level
                      level = np.random.randint(1, self.network_level + 1) if randomize else self.network_level
                      self.network_layer[row, col] = level
                      total_spawned_resources += level
                      network_count += 1
                      
           if attempts >= 2000:
                       print("Warning: Max attempts reached. Resources may be insufficient.")
           
                       
           # Exit condition: ensure all requirements are satisfied
           if (total_spawned_resources >= total_required_resources and
               storage_count >= self.num_storage and
               network_count >= self.num_network):
               break
   
            # Block cells with agents
            for a in self.players:
                if a.position and row == a.position[0] and col == a.position[1]:
                    return False
        
            return True
       
        
       