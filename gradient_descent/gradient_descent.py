from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy
from random import randint

color_norm = colors.Normalize(vmin=0, vmax=10)
scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')


def gradient_descent(w, h, y, iter, plot):
    orss = None
    alpha = 1.0
    c1 = 0
    ht = h.transpose()
    N = h.shape[0]

    for i in range(1, iter):
        # compute new W
        delta = y - (h * w)
        gdelta = ht * delta
        gdelta *= 2
        gdelta *= alpha
        gdelta /= N
        nw = w + gdelta

        # compute new RSS
        delta = y - (h * nw)
        rss = (delta * delta.transpose()).sum()
        if plt is not None:
            plot(nw, i)
        d = 0
        if orss is not None:
            d = rss - orss
        print "Iter:{0}, NW: {1}, RSS: {2}, Alpha: {3}, DRss: {4}".format(i, nw.flatten(), rss, alpha, d)

        # ignore new W is rss grow up or no change
        # and change alpha
        if orss is not None:
            if c1 <= 25:
                if orss < rss:
                    alpha *= 0.5
                    c1 += 1
                    print "rss no change, change alpha, c1: {0}".format(c1)
                    continue
                elif orss == rss:
                    alpha *= 0.5
                    c1 += 1
                    print "rss no change, change alpha, c1: {0}".format(c1)
                    continue
            else:
                print 'rss no change after change alpha'
                break
        c1 = 0
        orss = rss
        w = nw

    print '{0}'.format(w.flatten())
    return w




def DoMode_1(size, iter):
    def plot2d(x, h, w, i):
        plt.plot(x, numpy.asarray(h * w), '--', label='P_' + str(i))

    # 1- Generate Data
    X = numpy.matrix([[randint(0, 100)] for i in xrange(size)])
    H = numpy.ones((X.shape[0], X.shape[1] + 1))
    H[:, 1:] = X
    Y = H * numpy.matrix([[100], [3]])
    SW = numpy.matrix([[0], [0]])

    # Plot
    x0 = numpy.asarray(X[:, 0])

    # Gradient Descent
    W = gradient_descent(SW, H, Y, iter, lambda w, i: plot2d(x0, H, w, i))

    # Plot
    plt.plot(x0, numpy.asarray(Y), 'ro', label='P')
    plt.xlabel('X')
    plot2d(x0, H, W, -1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def DoMode_2(size, iter):
    def plot3d(a, x0, x1, h, w, i):
        a.scatter(x0, x1, numpy.asarray((h * w).astype(int)), marker='^', c=scalar_map.to_rgba(i), label='P_' + str(i))

    #2- Generate Data
    X = numpy.matrix([[randint(0, 100) for j in xrange(2)] for i in xrange(size)])
    H = numpy.ones((X.shape[0], X.shape[1] + 1))
    H[:, 1:] = X
    Y = H * numpy.matrix([[100], [3], [4]])
    SW = numpy.matrix([[0], [0], [0]])

    # Plot
    x0 = numpy.asarray(X[:,0])
    x1 = numpy.asarray(X[:,1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Gradient Descent
    W = gradient_descent(SW, H, Y, iter, lambda w, i: plot3d(ax, x0, x1, H, w, i))

    #plot
    ax.scatter(x0, x1, numpy.asarray(Y), marker='o', c='r', label='P')
    plot3d(ax, x0, x1, H, W, -1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()






# Parameter
Mode = 1 # 1 or 2

if Mode == 1:
    DataSetSize = 50
    IterationCount = 100
    DoMode_1(DataSetSize, IterationCount)
elif Mode == 2:
    DataSetSize = 250
    IterationCount = 100
    DoMode_2(DataSetSize, IterationCount)

