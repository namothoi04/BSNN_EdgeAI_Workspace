class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        print("Animal makes a sound")

class Dog(Animal): # Dog kế thừa từ Animal
    def __init__(self, name):
        super().__init__(name) # Gọi hàm khởi tạo của lớp cha (Animal)
        print("Dog is ready!")
        print(f"his name is {name}")
    def speak(self):
        print("Woof! Woof!")

dog = Dog("John")
dog.name