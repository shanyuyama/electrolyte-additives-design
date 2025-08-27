import numpy as np
import pandas as pd

def get_group_count():
    add_ls = []
    for i in range(len(gp_ls)):
        for j in range(len(gp_ls)):
            for k in range(len(gp_ls)):
                add_ls.append(gp_ls[i] + gp_ls[j] + gp_ls[k])
    return add_ls

def sort_gp(add_ls):
    i = 0
    while i < len(add_ls):
        # print(str(list(reversed(add_ls[i]))))
        if i > len(add_ls)-1:
            break
        ls = list(reversed(add_ls[i]))
        string = ls[0]+ls[1]+ls[2]
        if string in add_ls and string!= add_ls[i]:
            add_ls.remove(add_ls[i])
            i = i-1
        i = i+1
    return add_ls

def main():
    global gp_ls
    gp_ls = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o']
    add_ls = get_group_count()
    add_ls = sort_gp(add_ls)
    print(add_ls)
    print(len(add_ls))

if __name__ == '__main__':
    main()