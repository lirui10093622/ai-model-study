class MyClass:

    """A simple example class"""

    name = "MyClass1"

    def __init__(self):
        self.name = "MyClass"
        self.__doc__ = "A simple example object"

    @staticmethod
    def func(self):
        return 'hello world'

    def sayHello(self):
        print("hello")
