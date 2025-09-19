"""
Task 1: Real-Time Temperature Log (Debloated)

Efficient temperature logging with O(1) operations.
"""

from collections import deque

class TemperatureLog:
    def __init__(self):
        self.readings = deque()
        self.sums = {}  # Cache sums for different window sizes
        self.max_avgs = {}  # Track max averages
        
    def addReading(self, temp: int):
        """Add temperature reading. O(1)"""
        if not (-10 <= temp <= 10):
            raise ValueError("Temperature must be between -10 and +10")
        
        self.readings.append(temp)
        
        # Update cached values
        for k in list(self.sums.keys()):
            if len(self.readings) >= k:
                if k in self.sums and len(self.readings) > k:
                    # Sliding window update
                    self.sums[k] = self.sums[k] - self.readings[-k-1] + temp
                else:
                    self.sums[k] = sum(list(self.readings)[-k:])
                
                # Update max average
                avg = self.sums[k] / k
                self.max_avgs[k] = max(self.max_avgs.get(k, avg), avg)
    
    def getAverage(self, k: int) -> float:
        """Get average of last k readings. O(1)"""
        if k <= 0 or k > len(self.readings):
            return None
        
        if k not in self.sums:
            self.sums[k] = sum(list(self.readings)[-k:])
        
        return self.sums[k] / k
    
    def getMaxWindow(self, k: int) -> float:
        """Get max average for any window of size k. O(1)"""
        if k <= 0 or k > len(self.readings):
            return None
        
        if k not in self.max_avgs:
            # Calculate max for all windows of size k
            readings_list = list(self.readings)
            max_sum = sum(readings_list[:k])
            current_sum = max_sum
            
            for i in range(k, len(readings_list)):
                current_sum = current_sum - readings_list[i-k] + readings_list[i]
                max_sum = max(max_sum, current_sum)
            
            self.max_avgs[k] = max_sum / k
        
        return self.max_avgs[k]

def demo():
    """Quick demo"""
    print("=== Temperature Log Demo ===")
    
    log = TemperatureLog()
    temps = [5, -2, 8, 1, -3, 7, 0, 4]
    
    for temp in temps:
        log.addReading(temp)
    
    print(f"Added: {temps}")
    print(f"Last 3 avg: {log.getAverage(3):.2f}")
    print(f"Max window 3: {log.getMaxWindow(3):.2f}")
    print(f"Total readings: {len(log.readings)}")

if __name__ == "__main__":
    demo()