import numpy as np
import matplotlib.pyplot as plt


def plot_progression(Y, C):
    plt.figure()
    for k in range(Y.shape[0]):   #Y.shape = K, 2, I,   Y.shape[0] = K
        show_dots(Y[k,:,:],C)     #(x,y)-koord. med label
        plt.show()


def plot_model(model, Ys, C, n):
    grid, coordinates = get_grid_and_stuff(Ys, n)

    Z = model.fast_forward(grid)
    l = np.linspace(0,1,8)
    l = np.array([shading(shading2(x)) for x in l])

    plot_contours(*coordinates, Z, l, Ys, C)


def plot_separation(model, Ys, C, n):
    grid, coordinates = get_grid_and_stuff(Ys, n)

    Z = model.fast_landscape(grid)
    l = np.linspace(0,1,500)

    plot_contours(*coordinates, Z, l, Ys, C)


######## Internals

def show_dots(positions, labels):    #(Y0.shape, C)
    '''Visualize the output of get_data_spiral_2d'''
    plt.scatter(x=positions[0,:], y=positions[1,:], s=1, c=labels, cmap='bwr')  #c=color=label = {1,0}
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.axis('square')


def shading(x):
    if x == 0.0:
        return 0.0
    return 0.5 * np.tanh(np.tan(x * np.pi + np.pi / 2.0)) + 0.5


def shading2(x):
    if x < 0.5:
        return 0.5 - np.sqrt(0.25 - x**2)
    else:
        return 0.5 + np.sqrt(0.25 -(x-1.0)**2)

def get_box(Ys):          #Ys det (s/k)'te laget i Y
    xmin = min(Ys[0,:])   #Minimumverdi i den 0'te raden
    xmax = max(Ys[0,:])   #Maksimumsverdi i den 0'te raden
    xdelta = xmax-xmin    #Dersom xdelta >1 vil punktene øke avstand raskt. Dersom xdelta <1 vil punktene øke avstand saakte. Tilsv. for y
    xmin -= 0.2*xdelta
    xmax += 0.2*xdelta
    ymin = min(Ys[1,:])
    ymax = max(Ys[1,:])
    ydelta = ymax-ymin
    ymin -= 0.2*ydelta
    ymax += 0.2*ydelta
    return xmin, xmax, ymin, ymax

    
def get_grid(xcoordinates, ycoordinates):
    xv, yv = np.meshgrid(xcoordinates, ycoordinates)
    xs = xv.reshape(-1)
    ys= yv.reshape(-1)
    grid = np.stack([xs,ys])
    return grid


def get_grid_and_stuff(Ys, n):
    xmin, xmax, ymin, ymax = get_box(Ys)
    xcoordinates = np.linspace(xmin, xmax, n)
    ycoordinates = np.linspace(ymin, ymax, n)
    grid = get_grid(xcoordinates, ycoordinates)
    coordinates = ([xmin, xmax, ymin, ymax], xcoordinates, ycoordinates)
    return grid, coordinates


def plot_contours(box, xcoordinates, ycoordinates, Z, l, Ys, C1):
    n = xcoordinates.size
    plt.contourf(xcoordinates, ycoordinates, Z.reshape((n,n)), cmap='seismic', levels=l)
    plt.contour(xcoordinates, ycoordinates, Z.reshape((n,n)), levels=1, colors='k')
    plt.scatter(x=Ys[0,:], y=Ys[1,:], s=1, c=C1, cmap='bwr')
    plt.axis(box)
    plt.axis('equal')
    plt.show()

