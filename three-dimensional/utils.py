import torch
import numpy as np
import h5py
from lxml import etree
import matplotlib.pyplot as plt
from densNN import fourier_map
from matplotlib import cm, colors
from skimage import measure
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plotTO(x, y, z, X, Y, Z, rho_model, B, epoch):   
    rho = rho_model(fourier_map(x, y, z, B))
    ms_rho = rho.data.cpu().numpy().reshape(X.shape)
    density = ms_rho
    nely, nelx, nelz = X.shape
    nelm = max(nely,nelx,nelz)
    padding = np.zeros([nely+2,nelx+2,nelz+2])
    padding[1:-1, 1:-1, 1:-1] = np.copy(ms_rho)
    ms_rho = padding
    verts, faces, normals, values = measure.marching_cubes(ms_rho, 0.49) #set the density cutoff
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ls = LightSource(azdeg=30, altdeg=60)
    f_coord = np.take(verts, faces,axis = 0)
    f_norm = np.cross(f_coord[:,2] - f_coord[:,0], f_coord[:,1] - f_coord[:,0])
    cl = ls.shade_normals(f_norm)
    norm = colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.gray_r)
    rgb = mapper.to_rgba(cl).reshape([-1,4])
    mesh = Poly3DCollection(np.take(verts, faces,axis = 0)/nelx * (np.array([[nelm/nely, nelm/nelm, nelm/nelz]])))
    ax.add_collection3d(mesh)
    mesh.set_facecolors(rgb)
    ax.view_init(30, -60,vertical_axis='x')
    ax.set_box_aspect(aspect = (nelx,nelz,nely))
    ax.set_axis_off()
    plt.show()
    assert X.shape == Y.shape == Z.shape == density.shape, "Shapes of X, Y, Z, and density must match."   
    h5_filename = f'epoch_{epoch}.h5'
    xdmf_filename = f'epoch_{epoch}.xdmf'
    with h5py.File(h5_filename, 'w') as h5file:
        h5file.create_dataset('X', data=X)
        h5file.create_dataset('Y', data=Y)
        h5file.create_dataset('Z', data=Z)
        h5file.create_dataset('Density', data=density)
    # Write XDMF file
    root = etree.Element('Xdmf', Version='3.0')
    domain = etree.SubElement(root, 'Domain')
    grid = etree.SubElement(domain, 'Grid', Name='grid1', GridType='Uniform')
    topology = etree.SubElement(grid, 'Topology', TopologyType='3DRectMesh', Dimensions=f"{X.shape[0]} {X.shape[1]} {X.shape[2]}")
    geometry = etree.SubElement(grid, 'Geometry', GeometryType='ORIGIN_DXDYDZ')
    origin = etree.SubElement(geometry, 'DataItem', Dimensions="3", NumberType="Float", Precision="8", Format="XML")
    origin.text = f"{X[0,0,0]} {Y[0,0,0]} {Z[0,0,0]}"
    spacing = etree.SubElement(geometry, 'DataItem', Dimensions="3", NumberType="Float", Precision="8", Format="XML")
    spacing.text = f"{X[1,1,1]-X[0,0,0]}  {Y[1,1,1]-Y[0,0,0]}  {Z[1,1,1]-Z[0,0,0]}"
    attribute = etree.SubElement(grid, 'Attribute', Name='Density', AttributeType='Scalar', Center='Node')
    data_item = etree.SubElement(attribute, 'DataItem', Dimensions=f"{X.shape[0]} {X.shape[1]} {X.shape[2]}", NumberType="Float", Precision="4", Format="HDF")
    data_item.text = f"{h5_filename}:/Density"  
    # Save XDMF file
    tree = etree.ElementTree(root)
    tree.write(xdmf_filename, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    print(f"XDMF file '{xdmf_filename}' created successfully.")