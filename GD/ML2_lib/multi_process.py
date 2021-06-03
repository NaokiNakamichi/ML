from multiprocessing import Pool

class Foo:
    def __init__(self):
        pass

    def f(self,x):
        return x*x

    def re(self):
        print("roo")

        if __name__ == '__main__':
            with Pool(5) as p:
                tmp = p.map(self.f, [1, 2, 3])
                print(tmp)
                return tmp
