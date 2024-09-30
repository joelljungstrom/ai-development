# Exercises Week 2

from functools import reduce
import random
import itertools
import math
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import re
from collections import Counter


# Excersise 1
'''
class Person:
    def __init__(self, name, age):
        self.name = name 
        self.age = age
    
    def introduce(self):
        return f"Hej, jag heter {self.name} och är {self.age} år gammal."

person1 = Person("Sam", 29)
person2 = Person("Joel", 27)

print(person1.introduce())
print(person2.introduce())
'''
'''
# Exercise 2
class Person:
    def __init__(self, name, age, hobbies=[]):
        self.name = name 
        self.age = age
        self.hobbies = hobbies
    
    def introduce(self):
        return f"Hej, jag heter {self.name} och är {self.age} år gammal. Jag gillar {self.hobbies}"
    
    def add_hobby(self, hobby):
        self.hobby.append(hobby)
    
    def get_hobbies(self):
        return(list(map(str, self.hobbies)))
    
    def __str__(self):
        return f"Hej! Jag heter {self.name}. Jag är {self.age} år gammal och gillar {self.hobbies}"


person1 = Person("Sam", 29, "resor")
#person2 = Person("Joel", 27)

print(person1)
#print(person2.introduce())
'''
'''
# Exercise 3
class Bankkonto:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        self.amount = amount
        self.balance += amount
    
    def withdraw(self, amount):
        self.amount = amount
        if self.balance - amount < 0:
            return(print("Insufficient funds!"))
        else: 
            self.balance -= amount
    
    def display_amount(self):
        return f"Your current account balance is {self.balance}"

accountholder1 = Bankkonto("Joel", 0)
print(accountholder1.display_amount())
accountholder1.deposit(50)
print(accountholder1.display_amount())
accountholder1.withdraw(75)
accountholder1.withdraw(45)
print(accountholder1.display_amount())
'''
'''
# Exercise 4
class Matte:
    def __init__(self, a, b, r, d):
        self.a = a 
        self.b = b 
        self.r = r
        self.d = d
    
    def add(self):
        return self.a + self.b
    
    def subtract(self):
        return self.a - self.b
    
    def divide(self):
        return self.a / self.b
    
    def multiply(self):
        return self.a * self.b 
    
    def gcd(self):
        if self.b == 0:
            return self.a
        else:
            return self.a % self.b
    
    def area_circle(self):
        return math.pi * self.r ** 2
    
    def circumference(self):
        return math.pi * self.d
    
matte1=Matte(5, 4, 2, 8)
matte2=Matte(40, 60, 5, 10)
print(matte1.add())
print(matte2.gcd())
print(matte2.area_circle())
print(matte2.circumference())
'''
'''
# Exercise 5
with open("example.txt", "w") as file:
    file.write("Hello, world\n")
    file.write("This is an example\n")

with open("example.txt", "r") as file:
    print(file.read())

with open("example.txt", "a") as file:
    file.write("Adding some text\n")

with open("example.txt", "r") as file:
    print(file.read())
'''
'''
# Exercise 6
with open("example.txt", "w") as file:
    file.write("SAAB is known for its innovative engineering, and SAAB’s commitment to safety and performance makes SAAB a beloved brand among car enthusiasts who appreciate the distinctive design and reliability of SAAB vehicles.")

with open("example.txt", "r") as file:
    sentence = file.read().lower()
    words = sentence.split()
    counts = {}

    def count_words(words):
        for word in words:
            word = word.strip(",.")
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        return counts
    
    word_counts = count_words(words)

    print(word_counts)
'''
'''
# Exercise 7
class Contact:
    def __init__(self, name, phone, email):
        self.name = name 
        self.phone = phone 
        self.email = email
    
    def __str__(self):
        return f"{self.name}: {self.phone}, {self.email}"

class ContactBook:
    def __init__(self, contacts=[]):
        self.contacts = contacts
    
    def add_contact(self, contact):
        self.contacts.append(contact)
    
    def remove_contact(self, contact):
        self.contacts.remove(contact)
    
    def modify_contact(self, old_contact, new_name, new_phone, new_email):
        for contact in self.contacts:
            if contact == old_contact:  # Check if the contact object is the same
                contact.name = new_name 
                contact.phone = new_phone 
                contact.email = new_email
                return  # Exit after modifying the contact
        print("Contact not found.")
    
    def view_contacts(self):
        return(list(map(str, self.contacts)))

contact1 = Contact("Sam", "123-456-7890", "sam@example.com")
contact2 = Contact("Joel", "987-654-3210", "joel@example.com")

contact_book = ContactBook()
contact_book.add_contact(contact1)
contact_book.add_contact(contact2)
print("Before modification:\n")
print(contact_book.view_contacts())
contact_book.modify_contact(contact2, new_name="Joel", new_phone="079-339-9147", new_email="joel@ljungstrom.me")
print("After modification:\n")
print(contact_book.view_contacts())
'''
'''
# Exercise 8
class FileManager:
    def __init__(self):
        pass
    
    def read_file(self, filename):
        if os.path.exists(filename):
            with open(filename, "r") as file:
                print(file.read())
        else:
            print("File doesn't exist.")
    
    def write_file(self, filename, content):
        with open(filename, "w") as file:
            file.write(content)
    
    def append_file(self, filename, content):
        with open(filename, "a") as file:
            file.write(content)
    
    def delete_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print("File doesn't exist.")

file_manager = FileManager()
file_manager.write_file(filename="example.txt", content="SAAB cars")
file_manager.read_file("example.txt")
file_manager.append_file("example.txt", "\nWrite something")
file_manager.read_file("example.txt")
file_manager.delete_file("example.txt")
file_manager.read_file("example.txt")
'''
'''
# Exercise 9
class Stack:
    def __init__(self):
        self.items = []

    def __str__(self):
        return '. '.join(map(str, self.items))

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Kan inte poppa en tom stack.")
        return self.items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Kan inte visa en tom stack.")
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
stack=Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.is_empty())
print(stack.peek())
print(stack.pop())
print(stack)
print(stack.pop())
print(stack.pop())
print(stack.is_empty())
'''
'''
# Exercise 10
class Item:
    def __init__(self, item, amount, completed):
        self.item = item
        self.amount = amount
        self.is_completed = completed
    
    def __str__(self):
        return f"{self.amount} {self.item}. Is bought?: {self.is_completed}"

    def complete(self):
        if not self.is_completed:
            self.is_completed = True 
            return True 
        else:
            return False
    
    def return_item(self):
        self.is_completed = False

class ShoppingList: 
    def __init__(self, items=[]):
        self.items = items
        self.name = "Shopping List: "

    def add_item(self, item):
        self.items.append(item)
    
    def view_all_items(self):
        return(list(map(str, self.items)))
    
    def find_item(self, search_item):
        return next((item for item in self.items if item.item.lower() == search_item.lower()), None)

    def view_remaining_items(self):
        return list(filter(lambda item: not item.is_completed, self.items))
    
shopping_list = ShoppingList()
shopping_list.add_item(Item("Apples", 6, False))
shopping_list.add_item(Item("Bananas", 12, False))
shopping_list.add_item(Item("Carrots", 5, False))
shopping_list.add_item(Item("Eggs", 12, False))
shopping_list.add_item(Item("Milk", 2, False))
shopping_list.add_item(Item("Bread", 1, False))
shopping_list.add_item(Item("Chicken Breasts", 4, False))
shopping_list.add_item(Item("Rice", 1, False))
shopping_list.add_item(Item("Tomatoes", 8, False))
shopping_list.add_item(Item("Cheese", 500, False))
shopping_list.add_item(Item("Oranges", 5, False))

purchased_item = shopping_list.find_item("Apples")
if purchased_item:
    if purchased_item.complete():
        print(f"{purchased_item}.")
    else:
        print(f"{purchased_item} is already bought.")

remaining_items = shopping_list.view_remaining_items()
for item in remaining_items:
    print(item)
'''
'''
# Exercise 11
class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound 
    
    def make_sound(self):
        return self.sound

class Dog(Animal):
    def __init__(self, name):
        super().__init__(name, "Woof!")

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name, "Meow!")

class Cow(Animal):
    def __init__(self, name):
        super().__init__(name, "Moo!")

def animal_chorus(animals):
    for animal in animals:
        print(f"{animal.name} says {animal.make_sound()}")

dog = Dog("Buddy")
cat = Cat("Whiskers")
cow = Cow("Bessie")

animals = [dog, cat, cow]

animal_chorus(animals)
'''
'''
# Exercise 12
class GeometricShape:
    def __init__(self, name):
        self.name = name

    def area(self):
        # Implementera en metod som returnerar formens area
        pass

    def perimeter(self):
        # Implementera en metod som returnerar formens omkrets
        pass

    def __str__(self):
        # Returnera en sträng som beskriver formen
        return f"{self.name}. Area: {self.area()}, Shape: {self.perimeter()}"

class Rectangle(GeometricShape):
    def __init__(self, width, height):
        # Implementera konstruktorn
        super().__init__("Rectangle")
        self.width = width
        self.height = height

    # Implementera area() och perimeter() för Rectangle
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * self.width * self.height

class Circle(GeometricShape):
    def __init__(self, radius):
        # Implementera konstruktorn
        super().__init__("Circle")
        self.radius = radius

    # Implementera area() och perimeter() för Circle
    def area(self):
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        return self.radius * 2 * math.pi

# Skapa några instanser av Rectangle och Circle och testa dina metoder
rectangle = Rectangle(5, 10)
circle = Circle(5)

print(rectangle)
print(circle)
'''
'''
# Exercise 13
# 13.1
squares = [x ** 2 for x in range(1,11)]

# 13.2
even_numbers = [x for x in range (1,11) if x % 2 == 0]

# 13.3
sentence = "SAAB is known for its innovative engineering, and SAAB’s commitment to safety and performance makes SAAB a beloved brand among car enthusiasts who appreciate the distinctive design and reliability of SAAB vehicles."
word_lengths = [len(word) for word in sentence.split()]

print(squares)
print(even_numbers)
print(word_lengths)
'''
'''
# Exercise 14
# 14.1
def double(x):
    return x * 2

# 14.2
print(list(map(double, range(1,11))))

# 14.3
def is_even(x):
    if x % 2 == 0:
        return True 
    else:
        return False 

# 14.4
print(list(filter(is_even, range(1,11))))
'''
'''
# Exercise 15
# 15.1
numbers = [1,2,3,4,5,6,7,8,9,10]
def is_even(x):
    return x % 2 == 0

def quadratics(x):
    return x ** 2

print(list(map(quadratics, filter(is_even, numbers))))

# 15.2
def is_prime(n):
    for i in range(2,n):
        if n % i == 0:
            return False 
        else:
            return True

print(list(filter(is_prime, numbers)))
'''
'''
# Exercise 16
numbers = list(range(1,11))
print(reduce(lambda a, b: a*b, numbers))
print(reduce(lambda a, b: a if a > b else b, numbers))
'''
'''
# Exercise 17
# 17.1
number = 7
square = lambda x: x ** 2
print(square(number))

# 17.2
fruit_tuple = (
    ("Apple", 99),
    ("Banana", 25),
    ("Cherry", 4),
    ("Date", 52),
    ("Elderberry", 77)
)

fruits_sorted = sorted(fruit_tuple, key=lambda x: x[1])

print(fruit_tuple)
print(fruits_sorted)

# 17.3
#numbers = random_numbers = [random.randint(-50, 50) for _ in range(10)]
numbers = [6, -18, 40, 42, 46, 4, -22, -22, -33, -30]
positive_numbers = list(filter(lambda x: x if x > 0 else 0, numbers))

print(positive_numbers)
'''
'''
# Exercise 18
# 18.1
numbers=[]

for i in range(0,101):
    if i % 5 == 0 or i % 3 == 0:
        numbers.append(i)

print(numbers)

# 18.2
random_tuple = [(x,y) for x in range(5) for y in range(5)]
print(random_tuple)
'''
'''
# Exercise 19
x = input("Write a number: ")
y = input("Write a second number: ")

def division(x,y):
    try:
        x = int(x)
        y = int(y)
        division = x / y
        print(f"{x} divided by {y} is equal to {division}")
    except ZeroDivisionError:
        print("Not possible to divide by 0, try again.")
    except ValueError:
        print("Ensure to input numbers.")

division(x,y)
'''
'''
# Exercise 20
temperatures = [15.5, 16.0, 14.6, 11.9, 15.3, 16.2, 15.7]

plt.figure(figsize=(10, 5))
plt.plot(temperatures, marker='o', linestyle='-', color='b')

plt.title('Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')

plt.grid(True)

plt.show()
'''
'''
# Exercise 21
sales = pd.read_csv('ai-development/Exercises/week2/notes_and_data/sales_data.csv')

print("Top 5 rows:")
first_5 = sales.head()
print(first_5)

print("\nSales per category:")
sales_per_category = sales.groupby('Product')['SalesAmount'].sum()
print(sales_per_category)

print("\nAverage sale sper month:")
sales['Date'] = pd.to_datetime(sales['Date'])
sales['Month'] = sales['Date'].dt.to_period('M')
avg_sales_per_month = sales.groupby('Month')['SalesAmount'].mean().reset_index()
print(avg_sales_per_month)

print("\nBest sales day:")
best_day = sales.loc[sales["SalesAmount"].idxmax()].reset_index()
print(best_day)

print("Sales over time chart")
plt.figure(figsize=(10, 5))
plt.plot(sales['Date'], sales['SalesAmount'], marker='o', linestyle='-', color='b')

plt.title('Timeseries Sales Chart')
plt.xlabel('Day')
plt.ylabel('Daily Sales')
plt.grid(True)
plt.savefig('sales_chart.png', format='png', dpi=300)
plt.show()
'''
'''
# Exercise 22
x = np.random.randint(1, 10, size=(3, 3))
y = np.random.randint(1, 10, size=(3, 3))

z = np.dot(x,y)
print(z)

determinat = np.linalg.det(z)
print(determinat)
'''
'''
# Exercise 23
class FibonacciIterator:
    def __init__(self, max_value):
        self.max_value = max_value
        self.a, self.b = 0, 1
    
    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.a > self.max_value:
            raise StopIteration
        current = self.a 
        self.a, self.b = self.b, self.a + self.b 
        return current 

fibonacci_iterator = FibonacciIterator(50)

for number in fibonacci_iterator:
    print(number)
'''
'''
# Exercise 24
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Spara starttiden
        result = func(*args, **kwargs)  # Anropa den dekorerade funktionen
        end_time = time.time()  # Spara sluttiden
        execution_time = end_time - start_time  # Beräkna exekveringstiden
        print(f"Exekveringstid för {func.__name__}: {execution_time:.4f} sekunder")
        return result  # Returnera resultatet av funktionen
    return wrapper

@timer
def slow_function():
    time.sleep(2)

@timer
def quick_function():
    time.sleep(1)

# Anropa funktionerna
slow_function()
quick_function()
'''
'''
# Exercise 25
def fibonacci_generator(n):
    a = 0
    b = 1  # Initial values of the Fibonacci sequence
    for _ in range(n):
        yield a  # Yield the current Fibonacci number
        a = b
        b = a + b  # Update values for next Fibonacci number

# Example usage
num_terms = 10  # Specify how many terms you want
fib_gen = fibonacci_generator(num_terms)

print("Fibonacci sequence:")
for number in fib_gen:
    print(number)
'''
'''
# Exercise 26
class TextAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self.read_file()
        self.word_count = 0
        self.sentence_count = 0
        self.paragraph_count = 0
        self.common_words = []
        self.readability_index = 0

    def read_file(self):
        """Läs in textfilen och returnera innehållet."""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def analyze(self):
        """Analysera texten för att räkna ord, meningar och stycken."""
        self.word_count = len(re.findall(r'\b\w+\b', self.text))
        self.sentence_count = len(re.findall(r'[.!?]', self.text))
        self.paragraph_count = len(re.findall(r'\n\n+', self.text)) + 1  # Antal stycken

        # Räkna ord och hitta de vanligaste
        words = re.findall(r'\b\w+\b', self.text.lower())
        word_freq = Counter(words)
        self.common_words = word_freq.most_common(10)  # De 10 vanligaste orden

        # Beräkna läsbarhetsindex (Flesch-Kincaid)
        self.readability_index = self.calculate_readability_index()

    def calculate_readability_index(self):
        """Beräkna Flesch-Kincaid läsbarhetsindex."""
        total_syllables = sum(self.count_syllables(word) for word in re.findall(r'\b\w+\b', self.text))
        return (0.39 * (self.word_count / self.sentence_count)) + (11.8 * (total_syllables / self.word_count)) - 15.59

    def count_syllables(self, word):
        """Räkna antalet stavelser i ett ord."""
        word = word.lower()
        syllable_count = len(re.findall(r'[aeiouy]+', word))
        return max(syllable_count, 1)  # Minst 1 stavelse

    def generate_report(self):
        """Generera en rapport med analysresultat."""
        report = (
            f"Textanalysrapport för {self.file_path}\n"
            f"Ordantal: {self.word_count}\n"
            f"Meningar: {self.sentence_count}\n"
            f"Stycken: {self.paragraph_count}\n"
            f"Vanligaste orden: {self.common_words}\n"
            f"Läsbarhetsindex (Flesch-Kincaid): {self.readability_index:.2f}\n"
        )
        return report

# Exempelanvändning
file_path = '/Users/joel.ljungstroem/Documents/School/ITHS/iths/lib/python3.12/site-packages/numpy-2.1.1.dist-info/LICENSE.txt'  # Ange sökvägen till din textfil
analyzer = TextAnalyzer(file_path)
analyzer.analyze()
report = analyzer.generate_report()
print(report)
'''