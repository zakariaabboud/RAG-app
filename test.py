


 une fonction qui retourne la somme de deux nombres

def somme(a,b):#
    return a+b

# une fonction qui retourne la soustraction de deux nombres

def soustraction(a,b):
    return a-b

def factoriel(n):
    if n == 0:
        return 1
    else:
        return n*factoriel(n-1)
    
def puissance(a,n):
    if n == 0:
        return 1
    else:
        return a*puissance(a,n-1)
    
def puissance2(a,n):