"""
Mathieu Dehouck
Algorithmique et Programmation
2024-2025

The Art Of Parcouring Trees
"""
from random import seed, randint

# in this one we will create trees recursively and then we'll check things in them

# inductive definition of a tree of depth d
# [] is tree of depth 0
# if t1 and t2 are trees, [t1, t2] is a tree of depth 1 more than the max of depths of t1 and t2

# note how trees and parenthesed expressions are similar, in fact a well formed trees and well parenthesed expressions are isomorphic
# in some sense, they are equivalent, and computations that can be done on one can be done on the other and vice versa

seed(0)

def new_tree(depth):
    # depth is the maximum depth we will allow

    if depth == 0:
        return []

    else:
        side = randint(0,1) # choose which side is gonna be of depth d

        if side == 0:
            return [new_tree(depth-1), new_tree(randint(0, depth-1))]
        else:
            return [new_tree(randint(0, depth-1)), new_tree(depth-1)]

# a fun way of representing a tree to make it more palatable ?
# if the tree is big that does not help
def print_tree(tree, d):
    strs = ['' for _ in range(d+1)]
    base = str(tree)
    
    i = 0
    for c in base:
        if c == ']':
            i -= 1

        if c in '[] ':
            for j in range(d+1):
                if j == i:
                    strs[j] += c
                else:
                    strs[j] += ' '
        else:
            for j in range(d+1):
                if j == i-1:
                    strs[j] += c
                else:
                    strs[j] += ' '

        if c == '[':
            i += 1
        

    return strs

            

for d in range(10):
    t = new_tree(d)
    print(d, t, sep='\t')
    
    strs = print_tree(t, d)
    for s in strs:
        print(s)
    print()




# checking the depth of a tree

print('1.')

def depth(tree):
    if len(tree) == 0:
        return 0
    elif len(tree) == 2:
        return 1 + max(depth(tree[0]), depth(tree[1])) # the addition is out of the recursive call, count from bottom to top


def depth_alt(tree, current=0):
    if len(tree) == 0:
        return current
    elif len(tree) == 2:
        return max(depth_alt(tree[0], current+1), depth_alt(tree[1], current+1)) # the addition is inside the recursive call, count from top to bottom


for d in range(10):
    t = new_tree(d)
    print(d, depth(t), depth_alt(t), sep='\t') # note the answer is the same, hopefully



# your turn, if you can get the depth, you should be able to get the longest branch
print()
print('2.')

# in general there could be many branches of the same length, in fact there can be up to 2**k in a binary tree

# let modify the tree generation process to ensure only one branch has maximum length **    in fact it's exactly 2 because we have a binary tree, so if rrr is max then rrl is too

def new_tree_alt(depth):
    # depth is the maximum depth we will allow
    # we will also return the shape of the branch at the same time
    if depth == 0:
        return [], '.'

    else:
        side = randint(0,1) # choose which side is gonna be of depth d

        if side == 0:
            l, r = new_tree_alt(depth-1), new_tree_alt(randint(0, max(0,depth-2)))
            return [l[0], r[0]], 'l'+l[1] # the second branch is at least one shorter 
        else:
            l, r = new_tree_alt(randint(0, max(0,depth-2))), new_tree_alt(depth-1)
            return [l[0], r[0]], 'r'+r[1]


for d in range(10):
    t, branch = new_tree_alt(d)
    print(d, t, branch, sep='\t')


print()

# write you own function that get the longest branch recursively
# just as for the depth, the function can take one or two arguments and work from the top or the bottom

def longest(tree):
    NotImplemented
    return ''

def longest_alt(tree, current=''):
    NotImplemented
    return ''


for d in range(10):
    t, branch = new_tree_alt(d)
    l = longest(t)
    la = longest_alt(t)
    print(d, t, branch, l, la, branch[:-2] == l[:-2], branch[:-2] == la[:-2], sep='\t') # i ignore the last two characters of the branch so that rrl. and rrr. are considered equal, again, binary tree



# a bit of practice
print()
print('2.')

# try to solve these puzzles

expr = []
print(expr == [0, -1, 0, 1, 0, -1, 0, 1, 0, -1])


expr = []
print(expr == [30, 20, 12, 6, 2, 0])


expr = []
print(expr == [3, 15, 35, 63, 99])

expr = []
print(expr == [0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9])
