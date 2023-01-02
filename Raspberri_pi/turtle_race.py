#Hello World
from turtle import Turtle
from random import randint

laura = Turtle()

laura.color('red')
laura.shape('turtle')
laura.penup()
laura.goto(-160,100)
laura.pendown()

rik = Turtle()
lauren = Turtle()
carrie = Turtle()

rik.color('blue')
rik.shape('turtle')
rik.penup()
rik.goto(-160, 70)
rik.pendown()

lauren.color('purple')
lauren.shape('turtle')
lauren.penup()
lauren.goto(-160, 40)
lauren.pendown()

carrie.color('pink')
carrie.shape('turtle')
carrie.penup()
carrie.goto(-160, 10)
carrie.pendown()


for movement in range(120):
    laura.forward(randint(-5,11))
    rik.forward(randint(-10,16))
    lauren.forward(randint(-15,21))
    carrie.forward(randint(1,5))


print("Hello world")
