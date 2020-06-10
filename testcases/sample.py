# src/classes.py

class FooClass:
    def __init__(self, value):
        self.value = value

    def run(self):
        return self.value + 1

# src/functions.py

def run(val1, val2):

    foo = FooClass(value=val2)

    return val1 + foo.run()
