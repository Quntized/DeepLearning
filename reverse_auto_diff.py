from collections import defaultdict

class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

def add(a, b):
    value = a.value + b.value
    local_gradients = ((a, 1), (b, 1))
    return Variable(value, local_gradients)

def mul(a, b):
    value = a.value * b.value
    local_gradients = ((a, b.value), (b, a.value))
    return Variable(value, local_gradients)

def get_gradients(variable):

    gradients = defaultdict(lambda: 0)

    def compute_gradients(variable, path_value):
 
        for child_variable, local_gradient in variable.local_gradients:
            value_of_path_to_child = path_value * local_gradient
            gradients[child_variable] += value_of_path_to_child
            compute_gradients(child_variable, value_of_path_to_child)

    compute_gradients(variable, path_value=1)
    return gradients

a = Variable(4)
#print(a.local_gradients)
b = Variable(3)
#print(b.local_gradients)
c = add(a,b)
#print(c.local_gradients)
d = mul(a,c)
#print(d.local_gradients)
gradients = get_gradients(d)
#print(a.local_gradients)
gradients[a]