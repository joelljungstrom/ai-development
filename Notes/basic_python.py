# notes.py

# 1. Basic syntax, comments, printing
# Comment with a hastag or with tripple ''' ''' for multi-line comments.

print("Hello World!")

# 2. Variabler och datatyper
x = 5               # int
y = 3.14            # float
name = 'Python'     # string
is_fun = True       # boolean


# 3. Type checking, typkonvertering
print(type(x))
z = str(x) 
print(type(z))

a = "10"
b = a + "1"
print(b)
c = int(a) + 1
print(c)


# 4. String operations
print(len(name))
print(name.upper())
print(name.lower())
print("   whitespace_trash   ".strip())


# 5. String formatting
print(f"My name is {name} and I'm {int(a)+18} years old.")


# 6. Lists
fruits = ["apple", "banan", "kiwi", "apple"]
print(fruits[0])

fruits.append("pineapple")
print(fruits)


# 7. Dictionaries
person_dict1 = {"name":"Sam", "age":29, "city":"Cape Town"}
print(person_dict1["name"])
person_dict1["job"] = "Project Manager"
print(person_dict1)

person_list = []
person_dict2 = {"name":"Joel", "age":28, "city":"Båstad"}
person_list.append(person_dict1)
person_list.append(person_dict2)
print(person_list)


# 8. Sets
unique_fruits = set(fruits) # set removes any duplicates
print(unique_fruits)

unique_numbers = {1,2,3,4,5,5,5} # will automatically return distinct values
print(unique_numbers) 


# 9. Input from user
user_input = "joelljungstrom" #input("Please enter your user name: ")
print(f"Your user name is {user_input}.")


# 10. Conditionals, if statements
age = (int(a) * 3) - 3
if age >= 20:
    print("Du får dricka öl, och köpa på bolaget")
elif age >= 18:
    print("Du får dricka öl, men inte köpa på bolaget")
elif age >= 13:
    print("Du får inte dricka öl")
else:
    print("Du får verkligen inte dricka öl")

if user_input == "joelljungstrom":
    print("Hi joelljungstrom")
elif user_input != "joelljungstrom":
    print(f"Hi {user_input}")


# 11. Loops
# For loops
for fruit in fruits:
    print(fruit)

# While loops
count = 0 
while count < 10:
    print(count)
    count = count + 1

print("range loop")
for i in range(5):
    print(i)


# 12. Functions
def greet(name):
    #print(f"Hello, {name}")
    return f"Hello, {name}"

greeting = print(greet("Sam"))


