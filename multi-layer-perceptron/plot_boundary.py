import matplotlib.pyplot as plt
import numpy as np

def plot():
    xrange = numpy.arange(−4, 4, 0.1)
    yrange = numpy.arange(−4, 4, 0.1)
    xgrid, ygrid = numpy.meshgrid(xrange, yrange)

    npoints = xgrid.shape[0]∗xgrid.shape[1]
    xcoords = xgrid.reshape((npoints, 1))
    ycoords = ygrid.reshape((npoints, 1))
    samples = numpy.concatenate((xcoords, ycoords), axis = 1)

    ones = −numpy.ones(xcoords.shape)
    samples = numpy.concatenate((samples, ones), axis = 1)

    indicator = p.mlpfwd(samples)
    indicator = indicator.reshape(xgrid.shape)

    pylab.contour(xrange, yrange, indicator, (0.5, ))
    pylab.show()

if __name__ == '__main__':
    pass