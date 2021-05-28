class Test:
    def __iter__(self):
        n = 0
        while n < 10:
            yield n
            n = n + 1


test = Test()

for i, val in enumerate(test):
    print(val)
