# Native Python cumsum for 2 to 5 elements
# e.g.: cumsum(1,2,3) == (1,3,6)
def cumsum(a, b, c=None, d=None, e=None):
    if e is not None:
        s1 = a + b
        s2 = s1 + c
        s3 = s2 + d
        return (a, s1, s2, s3, s3 + e)
    elif d is not None:
        s1 = a + b
        s2 = s1 + c
        return (a, s1, s2, s2 + d)
    elif c is not None:
        s = a + b
        return (a, s, s + c)
    else:
        return (a, a + b)
