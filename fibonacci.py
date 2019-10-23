def Fibonacci(self, n):
        # write code here
        fibo_0 = 0
        fibo_1 = 1
        if n == 0:
            return fibo_0
        if n == 1:
            return fibo_1
        return self.Fibonacci(n-1)+ self.Fibonacci(n-2)