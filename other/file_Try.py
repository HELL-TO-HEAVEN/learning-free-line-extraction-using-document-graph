class Person:
    def __init__(self, first, last):
        self.first_name = first
        self.last_name = last

    def name(self):
        return self.first_name + " " + self.last_name

class Employee(Person):  # inheritance!
    def __init__(self, first, last, staff_number):
        # must execute super explicitly
        Person.__init__(self, first, last)
        self.staff_number = staff_number

    def get_employee(self):
        return self.name() + ", " + self.staff_number


x = Person("Marge", "Simpson")
y = Employee("Homer", "Simpson", "1007")

print(x.name())
print(y.get_employee())

