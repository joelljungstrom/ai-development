
from functools import reduce
import itertools

'''
bookCopy1 = {"title": "The Hobbit", "author": "J.R.R. Tolkien", "year": "1937", "genre": "Fantasy", "borrowed": 0}
bookCopy2 = {"title": "Harry Potter", "author": "J.K. Rowling", "year": "2001", "genre": "Fantasy", "borrowed": 1}
bookCopy3 = {"title": "To Kill a Mockingbird", "author": "Harper Lee", "year": "1960", "genre": "Fiction", "borrowed": 0}
bookCopy4 = {"title": "1984", "author": "George Orwell", "year": "1949", "genre": "Dystopian", "borrowed": 1}
bookCopy5 = {"title": "Pride and Prejudice", "author": "Jane Austen", "year": "1813", "genre": "Romance", "borrowed": 0}
bookCopy6 = {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": "1925", "genre": "Fiction", "borrowed": 1}
bookCopy7 = {"title": "Brave New World", "author": "Aldous Huxley", "year": "1932", "genre": "Dystopian", "borrowed": 0}
bookCopy8 = {"title": "The Catcher in the Rye", "author": "J.D. Salinger", "year": "1951", "genre": "Fiction", "borrowed": 1}
bookCopy9 = {"title": "The Alchemist", "author": "Paulo Coelho", "year": "1988", "genre": "Adventure", "borrowed": 0}
bookCopy10 = {"title": "The Picture of Dorian Gray", "author": "Oscar Wilde", "year": "1890", "genre": "Philosophical Fiction", "borrowed": 1}

library = [bookCopy1, bookCopy2, bookCopy3, bookCopy4, bookCopy5, bookCopy6, bookCopy7, bookCopy8, bookCopy9, bookCopy10]

for book in library:
    print(book["title"])
'''
# Bökigt att definera böcker enligt ovan. Kass struktur, istället kan man skapa classer

# Definera en ny class som hanterar böcker:
class Book:
    # Set up attributes for the class 'books':
    def __init__(self, title, author, year, genre, borrowed, pages):
        self.title = title
        self.author = author
        self.year = year 
        self.genre = genre 
        self.is_borrowed = borrowed
        self.pages = pages
    
    # Hantera print:
    def __str__(self):
        return f"{self.title} by {self.author} ({self.year})"
    
    # Skapa funktion för att låna boken:
    def borrow(self):
        if not self.is_borrowed:
            self.is_borrowed = True
            return True
        else:
            # Already borrowed
            return False
    
    # Skapa funktion för att lämna tillbaka boken:
    def return_book(self):
        self.is_borrowed = False

'''
# Skapa ett objekt (aka en bok):
bookObject1 = Book("The Hobbit", "J.R.R. Tolkien", 1937, "Fantasy", False)
bookObject2 = Book("Harry Potter", "J.K. Rowling", 2001, "Fantasy", True)
print(bookObject1)
print(bookObject2)

print(bookObject1.is_borrowed) # Printar statusen på is_borrowered för bookObject1
bookObject1.borrow() # <- triggar att vi lånar boken
bookObject1.return_book() # <- triggar att vi får tillbaka boken
print(bookObject1.is_borrowed)
'''

class Library:
    def __init__(self, books=[]):
        self.books = books
        self.name = "Kungliga Biblioteket" # definerar bibliotekets namn hårdkodat

    def add_book(self, book):
        self.books.append(book)
    
    def remove_book(self, book):
        self.books.remove(book)
    
    def find_book(self, title):
        return next((book for book in self.books if book.title.lower() == title.lower()), None)

    def list_books(self):
        return(list(map(str, self.books)))
    
    def available_books(self):
        return list(filter(lambda book: not book.is_borrowed, self.books))
    
    def get_total_pages(self):
        return reduce(lambda x, y: x + y, map(lambda book: book.pages, self.books), 0)
    
    def group_by_genre(self):
        return {genre: list(books) for genre, books in
                itertools.groupby(sorted(self.books, key=lambda x: x.genre), key=lambda x: x.genre)}
    
library = Library()
print(library.books)
print(library.name)

library.add_book(Book("To Kill a Mockingbird", "Harper Lee", 1960, "Fiction", False, 281))
library.add_book(Book("1984", "George Orwell", 1949, "Dystopian", True, 328))
library.add_book(Book("Pride and Prejudice", "Jane Austen", 1813, "Romance", False, 432))
library.add_book(Book("The Great Gatsby", "F. Scott Fitzgerald", 1925, "Fiction", True, 180))
library.add_book(Book("Brave New World", "Aldous Huxley", 1932, "Dystopian", False, 268))
library.add_book(Book("The Catcher in the Rye", "J.D. Salinger", 1951, "Fiction", True, 277))
library.add_book(Book("The Alchemist", "Paulo Coelho", 1988, "Adventure", False, 208))
library.add_book(Book("The Picture of Dorian Gray", "Oscar Wilde", 1890, "Philosophical Fiction", True, 254))

print("All books")
list(map(print, library.list_books()))

print("-------")
for book in library.books:
    print(book)

book_to_borrow = library.find_book("1984")
if book_to_borrow:
    if book_to_borrow.borrow():
        print(f"\nBorrowed: {book_to_borrow}.")
    else:
        print(f"\n{book_to_borrow} is already borrowed.")

print("\nAvailable books:")
for book in library.available_books():
    print(book)