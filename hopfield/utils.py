import numpy

x1 = numpy.array([-1, -1,  1, -1,  1, -1, -1,  1])
x2 = numpy.array([-1, -1, -1, -1, -1,  1, -1, -1])
x3 = numpy.array([-1,  1,  1, -1, -1,  1, -1,  1])

def rndPattern(n):
    "Create a random pattern of length n"
    return numpy.sign( numpy.random.randn(n) )

def learn(patterns):
    "Use Hebbs rule to find the weight matrix for a list of patterns"
    n = len(patterns[0])
    w = numpy.zeros((n, n))
    for p in patterns:
        w += numpy.outer(p, p)
    return w - numpy.diag(numpy.diag(w))

def update(w, x):
    "Apply the Hopfield update rule using weight matrix w on pattern x"
    return numpy.sign(numpy.dot(w, x))

def updateOne(w, x):
    "Update one element in x"
    i = numpy.random.randint(len(x))
    x[i] = numpy.sign(numpy.dot(w[i,:], x))

def energy(w, x):
    "Return the energy of state x, using weight matrix w"
    return -numpy.dot(x, numpy.dot(w, x))

def samePattern(x, y):
    "Check if patterns x and y are equal"
    return numpy.all(x == y)

def flipper(x, n):
    "Flip the values of n elements in pattern x"
    flip = numpy.array([-1]*n + [1]*(len(x)-n))
    numpy.random.shuffle(flip)
    return x * flip
