'''
There provides some helper functions to create the 3D plots in the paper.
'''

import matplotlib.pyplot as plt
from matplotlib import animation
from skimage import measure
from plotly.offline import init_notebook_mode, plot
from plotly.figure_factory import create_trisurf

init_notebook_mode(connected=True)

"----------------------------------------------------------"
" Visualization Function "
"----------------------------------------------------------"


def plot_voxels(voxels, aux=None):
    """ plot voxels stack
    """
    if aux is not None:
        assert voxels.shape == aux.shape
    n = voxels.shape[0]
    for i in range(n):
        plt.figure(figsize=(4, 4))
        plt.title("@%s" % i)
        plt.imshow(voxels[i], cmap=plt.cm.gray)
        if aux is not None:
            plt.imshow(aux[i], alpha=0.3)
        plt.show()


def plot_hist(voxel):
    """ plot histogram of image or voxel data
    """
    plt.hist(voxel.flatten(), bins=50, color='c')
    plt.xlabel("pixel value")
    plt.ylabel("frequency")
    plt.show()


def plot_voxels_stack(stack, rows=6, cols=6, start=10, interval=5):
    """ plot image stack for a scan
    """
    fig, ax = plt.subplots(rows, cols, figsize=[18, 18])
    for i in range(rows * cols):
        ind = start + i * interval
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    plt.show()


def plot_voxel(voxel, title='voxel'):
    """ plot voxel gray image
    """
    plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.imshow(voxel, cmap=plt.cm.gray)
    plt.show()


def plot_voxel_slice(voxels, slice_i=0, title="@"):
    plot_voxel(voxels[slice_i], title=title + str(slice_i))


def animate_voxels(voxels, aux=None, interval=300):
    """ plot voxels animation
    """
    fig = plt.figure()
    layer1 = plt.imshow(voxels[0], cmap=plt.cm.gray, animated=True)
    if aux is not None:
        assert voxels.shape == aux.shape
        layer2 = plt.imshow(aux[0] * 1., alpha=0.3, animated=True)

    def animate(i):
        plt.title("@%s" % i)
        layer1.set_array(voxels[i])
        if aux is not None:
            layer2.set_array(aux[i] * 1.)

    ani = animation.FuncAnimation(fig, animate, range(1, voxels.shape[0]),
                                  interval=interval, blit=True)
    return ani


# 3D mesh visualization

def make_mesh(image, threshold, step_size):
    """ aux function to make mesh for 3d plot, need be called first
    step_size at leat 1, lareger step_size lead to low resolution but low time consuming
    """
    # print("Transposing surface")
    p = image.transpose(2, 1, 0)

    # print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(
        p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces


def hidden_axis(ax, r):
    ax.showgrid = False
    ax.zeroline = False
    ax.showline = False
    ax.ticks = ''
    ax.showticklabels = False
    # ax.showaxeslabels = False
    ax.range = r
    ax.title = ""



def plotly_3d_to_html(verts, faces, filename="tmp.html", title="3d visualization", zyx_range=None):
    """ use plotly offline to plot 3d scan
    """
    x, y, z = zip(*verts)
    # print("Drawing")

    # Make the colormap single color since the axes are positional not intensity.
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    # colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    # fig = FF.create_trisurf(x=x,
    fig = create_trisurf(x=x,
                         y=y,
                         z=z,
                         showbackground=False,
                         plot_edges=False,
                         colormap=colormap,
                         simplices=faces,
                         #  backgroundcolor='rgb(240, 240, 240)',
                         title=title,
                         show_colorbar=False)
    if zyx_range is not None:
        hidden_axis(fig.layout.scene.zaxis, zyx_range[0])
        hidden_axis(fig.layout.scene.yaxis, zyx_range[1])
        hidden_axis(fig.layout.scene.xaxis, zyx_range[2])
        # fig.layout.scene.zaxis.range = zyx_range[0]
        # fig.layout.scene.yaxis.range = zyx_range[1]
        # fig.layout.scene.xaxis.range = zyx_range[2]
    plot(fig, filename=filename)
    return fig


def plotly_3d_scan_to_html(scan, filename, threshold=0.5, step_size=3, title="3d visualization", zyx_range=None):
    v, f = make_mesh(scan, threshold=threshold, step_size=step_size)
    return plotly_3d_to_html(v, f, filename=filename,
                             title=title, zyx_range=zyx_range)
