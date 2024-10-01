# week1_exercises.py

import random 


# 1. Skriv ett profram som tar emot en sträng som input och skriver ut längden på strängen
'''
string1 = input("Enter any text: ")
print(f"The length of the string is {len(string1)} characters.")
'''

# 2. Skriv ett program som skriver ut frekvensen av tecken i en given sträng.

def count_characters(s):
    character_counts = {}
    for char in s:
        if char in character_counts:
            character_counts[char] += 1
        else:
            character_counts[char] = 1
    return character_counts

string2 = input("Enter any text: ")
character_counts = count_characters(string2)
print(character_counts)


# 3. Skriv ett program som för en given sträng skriver ut de två första och de två sista tecknen i strängen (på valfritt format)
'''
string3 = input("Enter any text: ")
print(f"First 2 characters: {string3[0:2]}, last 2 characters: {string3[-2:]}")
'''

# 4. Skriv ett program som tar två strängar som input och skapar EN ny sträng där de två första tecken i varje sträng bytts ut.
'''
string4 = input("Type a word: ")
string5 = input("Type a word: ")
print(string5[0:2]+string4[-1]+" "+string4[0:2]+string5[-1])
'''

# 5. Skriv ett program som lägger till "ing" i slutet av en given sträng, om strängen är kortare än 3 tecken ska den lämnas ofärndrad.
'''
string6 = input("Type a word: ")
if len(string6) <= 3:
    print(string6)
else:
    print(string6+"ing")
'''

# 6. Skriv ett program som först tar bort all whitespace (mellanslag, tab (\t), newline(\n)), och sedan även tar bort alla tecken på ojämna indexvärden, från given sträng.
'''
string7 = input("Type a word: ")
string7 = string7.strip().replace(" ","")

print(string7[::2])
'''

# 7. Skriv ett program som tar en komma-separerad sekvens av ord och skriver ut de unika orden i alfabetisk ordning.
'''
def unique_words(w):
    library=[]
    for word in w.split(','):
        word = word.strip()
        if word not in library:
            library.append(word)
    return sorted(library)

print(unique_words(input("Type a list of words, comma-separated: ")))
'''

# 8. Skriv en funktion som konverterar en given sträng till versaler (uppercase) om den innehåller minst 2 versaler bland de 4 första tecknen.
'''
def contains_upper(s):
    upper_count=0
    for char in s:
        if char.isupper():
            upper_count += 1
    return upper_count

string8 = input("Type a word: ")
if contains_upper(string8[0:4]) >= 2:
    print(string8.upper())
else:
    print(string8)
'''

# 9. Skriv en funktion som vänder (reverse) på en sträng om dess längd är en multipel av 4.
'''
string9 = input("Type a word: ")

if len(string9) % 4 == 0:
    print(f"{''.join(reversed(string9))}")
else:
    print(string9)
'''

# 10. Skriv en funktion som skapar en ny sträng bestående av 4 kopior av de två sista tecken i en given sträng.
'''
string10 = input("Type a word: ")

string10 = string10[-2:]+string10[-2:]+string10[-2:]+string10[-2:]

print(string10)
'''

# 11. Skriv en funktion som tar emot en lista med ord och returnerar det längsta ordet samt dess längd
'''
def longest_word(w):
    words={}
    for word in w.split(','):
        word = word.strip()
        words[word] = len(word)
    max_char=max(words.values(), default=0)
    longest_word=[word for word, length in words.items() if length == max_char]
    return longest_word, max_char

string11 = input("Type a list of words, comma-separated: ")

string11 = longest_word(string11)

print(string11)
'''

# 12. Skriv ett program som genererar en enkel multiplikationsmodell för tal 1-10. Hur snyggt kan du få tabellen? Läs på om sträng-formattering i Python.
'''
x = list(range(1, 11))
y = list(range(1, 11))

# Print header
print("   ", end="")
for num in y:
    print(f"{num:4}", end="")  # Formatting for alignment
print()  # New line

# Print separator
print("    " + "----" * len(y))

# Print multiplication table
for i in x:
    print(f"{i:2} |", end="")  # Print row header
    for j in y:
        print(f"{i * j:4}", end="")  # Print multiplication result
    print()  # New line for the next row
'''

# 13. Skriv en funktion som beräknar fakulteten av ett givet tal
'''
number = input("Write any number: ")

def factorial(i):
    result=1
    if int(number):
        for i in range(1,int(number)+1):
            result *= i
    return result

try:
    number = int(number)
    if number < 0:
        print("Factorial not available for negative numbers.")
    else:
        print(f"Factorial of {number} is {factorial(number)}.")
except ValueError:
    print("Please enter a whole number.")
'''

# 14. Skapa ett enkelt gissningsspel där datorn väljer ett slumpmässigt tal mellan 1-100 (eller annat intervall), och låt användaren gissa tills de hittar rätt nummer. För varje felaktig gissning berättar datorn om det rätta svaret är högre eller lägre än spelarens gissning.
'''
j = random.randint(1,100)
i = None

while i != j:
    i = int(input("Guess the number between 1 and 100: "))
    if i == j:
        print("Correct number!")
    elif i < j:
        print("Number is higher than i.")
    elif i > j:
        print("Number is lower than i.")
'''

# 15. Skriv ett program som kontrollerar om ett givet ord är ett palindrom (läses likadant framifrån som bakifrån).
'''
def palindrom(word):
    reversed_word = ''.join(reversed(word))
    if reversed_word == word:
        print(f"The word {word} is a palindrom.")
    else:
        print(f"The word {word} is not a palindrom.")

word = input("Type a word to find out if it's an palindrom: ")

palindrom(word)
'''

# 16. Skriv ett python program som itererar mellan 1 och 50, 
    # *	om talet är delbart med 3 printar den "fizz"
    # *	om talet är delbart med 5 printar den "buzz", 
    # *	om talet är delbart med både 3 och 5 så printar den "FizzBuzz"
    # *	annars printar den bara ut talet
'''
j = random.randint(1,50)

def devisible(j):
    if j % 3 == 0 and j % 5 == 0:
        print("FizzBuzz")
    elif j % 3 == 0:
        print("fizz")
    elif j % 5 == 0:
        print("buzz")
    else:
        print(j)

devisible(j)
'''