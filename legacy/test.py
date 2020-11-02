class Obj:
    def __init__(self):
        self.s = set()
        self.a = 1
        self.b = 3

    def __str__(self):
        return f"{repr(self.s)} {self.a} {self.b}"
    
    def __repr__(self):
        return f"{repr(self.s)} {self.a} {self.b}"


d = {}
for i in range(10):
    a = Obj()
    d[i] = Obj()


for i in range(10):
    d[i].s.add(i)
    d[i].a += i
    d[i].b += i

print(d)


# print(objs)
