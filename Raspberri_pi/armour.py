# Hello world
# Armour

class Armour():

    def __init__(self, armour_type):
        self.name = armour_type
        self.description = None
        self.attributes = None
        self.linked_items = {}


    def get_description(self):
        return self.description
    
    def set_description(self, armour_description):
        self.description = armour_description

    def describe(self):
        print(self.description)

    def get_name(self):
        return self.name

    def link_item(self, item_to_link, combo_effect):
        self.linked_items[combo_effect] = item_to_link

    def get_details(self):
        print("You have ", self.name)
        print(self.description)
        print("")
        for combo_effect in self.linked_items:
            item = self.linked_items[combo_effect]
            print(" The ", item.get_name(), " gives you ", combo_effect)
        print("-------")

    def explore(self, combo_effect):
        if combo_effect in self.linked_items:
            return self.linked_items[combo_effect]
        else:
            print("This is not a valid effect.")
            return self

    

        
