


def a(x):
    y = x
    def b():
        print(y)
        # return y
    return b
print(a(1))
print(a())