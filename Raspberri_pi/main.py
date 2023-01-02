# Hello World

from room import Room
dining_hall = Room("dining_hall")
kitchen = Room("kitchen")
ballroom = Room("ballroom")
kitchen.set_description("A dank and dirty room buzzing with flies.")
kitchen.link_room(dining_hall, "south")


ballroom.set_description("A large room with ornate golden decorations on each wall")
ballroom.link_room(dining_hall, "east")



dining_hall.set_description("A vast room with shiny wooden floor; huge candlesticks guard the entrance")
dining_hall.link_room(kitchen, "north")
dining_hall.link_room(ballroom, "west")

#kitchen.get_description()
print("...   ...   ...")

current_room = kitchen

while True:
    print("\n")
    current_room.get_details()
    command = input("> ")
    current_room = current_room.move(command)




print("Hello world.")

