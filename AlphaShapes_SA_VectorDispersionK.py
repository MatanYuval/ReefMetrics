import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import open3d as o3d
from sys import setrecursionlimit
import matplotlib._color_data as mcd
import pandas as pd
import seaborn as sns
import win32evtlogutil
from scipy.stats import linregress as lr
from pathlib import Path
import os
import numpy as np
from numpy import random
import glob
import pandas as pd


mesh = o3d.io.read_triangle_mesh('E:/Downloads2/FinalExperiment/Box.ply')


models  = (glob.glob('E:/Downloads2/FinalExperiment/*.ply'))
os.listdir('E:/Downloads2/InnovationModels/')
resPath = "E:/Downloads2/InnovationModels/obj/"
    

def calculate_1_over_k(mesh):
    #mesh = o3d.io.read_triangle_mesh(mdl)
    mesh.compute_vertex_normals(normalized=True)
    cos = np.array(mesh.triangle_normals)
    R1 = ((cos.sum(0) ** 2).sum() ** 0.5)
    i = len(cos)
    k = (i - R1) / (i - 1)
    # print(f"scene {scene} {year}, 1/k={k}")
    return k
from scipy.stats import linregress as lr
slopeArry = []
nameArry = []
for mdl in models:
    
    mesh = o3d.io.read_triangle_mesh(mdl)
    k = calculate_1_over_k(mesh)
    name = mdl
    name = name.replace('E:/Downloads2/FinalExperiment\\', "")
    name = name.replace(".ply", "")
    slopeArry.append([name, k])
    print(mdl, k )

def AlphaShapes(mesh):
    rng = [0.025, 0.05, 0.10, 0.2, 0.4, 0.6]  # In alpha smoothing alpha = 0.1 i.e 10 cm
    # range = np.arange(0.1, 1.1, 0.1)
    arry = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    for alpha in rng:
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        sa = mesh.get_surface_area()
        print(f"SA={sa:.3f}")
        arry.append([alpha, sa])
    xs = [x[0] for x in arry]
    ys = [(y[1]) for y in arry]
    coeffs, _ = np.polynomial.polynomial.polyfit(np.log(xs), np.log(ys), 1, full=True)
#    ax = plt.figure()
  #  ax = sns.regplot((np.log(xs)), np.log(ys), fit_reg=True)
 #   slope, intercept, r_value, p_value, std_err = lr(x=ax.get_lines()[0].get_xdata(), y=ax.get_lines()[0].get_ydata())
    #slope = 2 - slope
    slope = 2 -coeffs[1]
    print(slope)
    return slope
kArry = []
slopeArry = []
for mdl in models:
    mesh = o3d.io.read_triangle_mesh(mdl)
    k = calculate_1_over_k(mesh)
    name = mdl
    name = name.replace('E:/Downloads2/FinalExperiment\\', "")
    name = name.replace(".ply", "")
    kArry.append([name, k])
    print(mdl, k)

for mdl in models:
    mesh = o3d.io.read_triangle_mesh(mdl)
    slope = AlphaShapes(mesh)
    name = mdl
    name = name.replace('E:/Downloads2/FinalExperiment\\', "")
    name = name.replace(".ply", "")
    slopeArry.append([name, slope])
    print(name, slope)
