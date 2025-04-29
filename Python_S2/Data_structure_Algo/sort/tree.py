def BinaryTree(r):
    return [r, [], []]

def InserLeft(newBranch,root):
    l = root.pop(1)
    if len(l) > 1:
        root.insert(1, [newBranch, l, []])
    else:
        root.insert(1,[newBranch, [], []])

    return r

def InserRight(newBranch,root):
    r = r.pop(2)
    if len(r) > 1:
        root.insert(2, [root,[],[newBranch, [], r]])
    else:
        root.insert(2, [root, [], [newBranch, [], []]])
    
    return root

def GetRoot(root):
    return root[0]

def SetRoot(root,value):
    root[0] = value

def GetLeft(root):
    return root[1]

def SetLeft(root,value):
    root[1] = value

def GetRight(root):
    return root[2]

def SetRight(root,value):
    root[2] = value

r = BinaryTree(3)
print(InserLeft(0,r))
print(InserLeft(1,r))
l = GetLeft(r)
print(l)
SetRoot(l,5)
print(r) # l referts to r[1], change l will also change r