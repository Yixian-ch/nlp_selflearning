class Stack:
    def __init__(self):
        self.item = []
    
    def push(self,ele):
        self.item.append(ele)
    
    def drop(self):
        return self.item.pop()
    
    def peek(self):
        return self.item[-1]
    
    def is_empty(self):
        return len(self.item) == 0
    
    def size(self):
        return len(self.item)