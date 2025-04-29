class BinaryTree():
    def __init__(self,root):
        self.root = root
        self.left = None
        self.right = None

    def InserLeft(self, NewBranch):
        if self.left == None:
            self.left = BinaryTree(NewBranch)
        else:
            t = BinaryTree(NewBranch)
            t.left = self.left
            self.left = t

    def InserRight(self, NewBranch):
         if self.right == None:
            self.right = BinaryTree(NewBranch)
         else:
            t = BinaryTree(NewBranch)
            t.right = self.right # package the current right into newbranch
            self.right = t # replace old right with new right that has packaged the old right

    def ShowLeft(self):
        return self.left
    
    def ShowRight(self):
        return self.right
    
    def SetRootValue(self,value):
        self.root = value
    
    def ShowRoot(self):
        return self.root
    
        
    def preorder(self):
        print(self.ShowRoot())
        if self.left:
            self.left.preorder()
        
        if self.right:
            self.right.preorder()

    def inorder(self):
        if self.left:
            self.left.inorder()
        print(self.ShowRoot())

        if self.right:
            self.right.inorder()

    def count_leaves(self):
        if self.left == None and self.right == None:
            return 1
        
        leaves = 0
        if self.left:
            leaves += self.left.count_leaves()
        if self.right:
            leaves += self.right.count_leaves()
        return leaves
    
    def count_depth(self):
        left_depth = self.left.count_depth() if self.left else 0
        right_depth = self.right.count_depth() if self.right else 0
        return max(left_depth, right_depth) + 1


tree = BinaryTree(1)
tree.InserLeft(2)
tree.InserRight(3)
tree.left.InserLeft(4)
tree.left.InserRight(5)
# tree.inorder()
print(tree.count_leaves())
print(tree.count_depth())