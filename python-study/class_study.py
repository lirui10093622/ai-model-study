import my_class as mc
from python_lang.my_class import MyClass

MyClass.x = "xxxx"
print(MyClass.__dict__)

del MyClass.x

myObj = MyClass()
myObj.sayHello()

myObj.xx = "xxx"
print(myObj.xx)
