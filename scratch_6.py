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
import open3d as o3d
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

import open3d as o3d
import glob

import pandas as pd

models  = (glob.glob('E:/Downloads2/FinalExperiment/*.ply'))
os.listdir('E:/Downloads2/InnovationModels/')
resPath = "E:/Downloads2/InnovationModels/obj/"

for mdl in models:
    print(models)
    mesh = o3d.io.read_triangle_mesh(models[6])
    name = mdl
    name = name.replace('E:/Downloads2/InnovationModels\\', "")
    name = name.replace("ply", "obj")
    o3d.io.write_triangle_mesh(resPath+"/"+name, mesh)
name = name.replace("ply", "obj")
files = (glob.glob('E:/Downloads2/InnovationModels/obj/*.txt'))
import pandas as pd
slopeArry = []
nameArry = []
for filee in files:
    print(filee)
    data = pd.read_table(filee, delimiter = ' ')
    data.columns = ['line_ID' ,'Dilation_Radius', 'log(Dilation_Radius)', 'log(Influence_Volume)']
    name = filee
    name = name.replace('E:/Downloads2/FinalExperiment/C5NR/obj\\', "")
    name = name.replace(".txt", "")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure()
    ax = sns.regplot(data['log(Dilation_Radius)'], data['log(Influence_Volume)'], fit_reg=True)
    slope, intercept, r_value, p_value, std_err = lr(x=ax.get_lines()[0].get_xdata(), y=ax.get_lines()[0].get_ydata())
    ax.set_ylabel('log(Influence_Volume)', fontsize=25)
    ax.set_xlabel('log(Dilation_Radius)', fontsize=25)
    slopee = 3 - slope
    title = ("model is %s" %name + "  Slope is 3- fit %1.3f " % slopee)
    ax.set_title(title,fontsize=25)
    slopeArry.append([name,  slopee])
resPath

for row in slopeArry:
    print(row)
    mesh = o3d.io.read_triangle_mesh(
        "C:/Users/USER/Downloads/DatasetForExperimentFinal-20210810T055959Z-001/DatasetForExperimentFinal/Kza5m2020.ply")

    # _, mesh = read_model(pcd_name, model_type)
    

def calculate_1_over_k(mesh):
    #mesh = o3d.io.read_triangle_mesh(mdl)
    mesh.compute_vertex_normals(normalized=True)
    cos = np.array(mesh.triangle_normals)
    R1 = ((cos.sum(0) ** 2).sum() ** 0.5)
    i = len(cos)
    k = (i - R1) / (i - 1)
    # print(f"scene {scene} {year}, 1/k={k}")
    return k

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

from scipy.stats import linregress as lr

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
    ax = plt.figure()
    ax = sns.regplot((np.log(xs)), np.log(ys), fit_reg=True)
    slope, intercept, r_value, p_value, std_err = lr(x=ax.get_lines()[0].get_xdata(), y=ax.get_lines()[0].get_ydata())
    #slope = 2 - slope
    slope = 2 -coeffs[1]
    print(slope)
    return slope
slopeArry = []


def Smooth_1_K(mesh):
    arry = []
    mesh_out = mesh
    k = calculate_1_over_k(mesh_out)
    arry.append([(0), (k)])
    for i in range(1,10,1):
        mesh_out = mesh_out.filter_smooth_simple(number_of_iterations=i)
        mesh_out.compute_vertex_normals()
        k = calculate_1_over_k(mesh_out)
        arry.append([(i+1), (k)])
    xs = [x[0] for x in arry]
    xs[0] = 1
    ys = [(y[1]) for y in arry]
    coeffs, _ = np.polynomial.polynomial.polyfit(np.log(xs), np.log(ys), 1, full=True)
    slope = 2 - coeffs[1]
    print(slope)
    return (slope)
slopeArry = []
for mdl in models:
    mesh = o3d.io.read_triangle_mesh(mdl)
    k = Smooth_1_K(mesh)
    name = mdl
    name = name.replace('E:/Downloads2/FinalExperiment\\', "")
    name = name.replace(".ply", "")
    slopeArry.append([name, k])
    print(mdl, k)

for mdl in models:
    mesh = o3d.io.read_triangle_mesh(mdl)
    k = calculate_1_over_k(mesh)
    name = mdl
    name = name.replace('E:/Downloads2/FinalExperiment\\', "")
    name = name.replace(".ply", "")
    slopeArry.append([name, k])
    print(mdl, k)

for mdl in models:

    mesh = o3d.io.read_triangle_mesh(mdl)
    slope = AlphaShapes(mesh)
    name = mdl
    name = name.replace('E:/Downloads2/FinalExperiment\\', "")
    name = name.replace(".ply", "")
    slopeArry.append([name, slope])
    print(name, slope)
for mdl in models[2]:
    print(mdl)
    mesh = o3d.io.read_triangle_mesh(models[2])
    hull,_ = mesh.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([mesh, hull_ls])
    a = hull.get_volume()
    name = mdl
    name = name.replace('E:/Downloads2/FinalExperiment\\', "")
    name = name.replace(".ply", "")
    print(name,",", a)
    #slopeArry.append([name, hull.get_volume])

def cHull_SA(mesh):
    hull, _ = mesh.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([mesh, hull_ls])
    #shelter_space = hull.get_volume() - mesh.get_volume()
    saRatio = (mesh.get_surface_area()/ hull.get_surface_area())
    return (saRatio)
slopeArry = []
for mdl in models:
    print(mdl)
    mesh = o3d.io.read_triangle_mesh(mdl)
    slope = cHull_SA(mesh)
    name = mdl
    name = name.replace('E:/Downloads2/FinalExperiment\\', "")
    name = name.replace(".ply", "")
    slopeArry.append([name, slope])

cHull_SA(mesh)


files = (glob.glob('E:/Downloads2/FinalExperiment/obj/*.txt'))

import pandas as pd
slopeArry = []
nameArry = []
for filee in files:
    print(filee)
    data = pd.read_table(filee, delimiter = ' ')
    data.columns = ['line_ID' ,'Dilation_Radius', 'log(Dilation_Radius)', 'log(Influence_Volume)']
    name = filee
    name = name.replace('E:/Downloads2/FinalExperiment/C5NR/obj\\', "")
    name = name.replace(".txt", "")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure()
    ax = sns.regplot(data['log(Dilation_Radius)'], data['log(Influence_Volume)'], fit_reg=True)
    slope, intercept, r_value, p_value, std_err = lr(x=ax.get_lines()[0].get_xdata(), y=ax.get_lines()[0].get_ydata())
    ax.set_ylabel('log(Influence_Volume)', fontsize=25)
    ax.set_xlabel('log(Dilation_Radius)', fontsize=25)
    slopee = 3 - slope
    title = ("model is %s" %name + "  Slope is 3- fit %1.3f " % slopee)
    ax.set_title(title,fontsize=25)
    slopeArry.append([name,  slopee])
data = pd.read_csv("E:/Downloads2/Results 2nd chapter - Sheet2.csv",  index_col= 8)
data = pd.read_csv("E:/Downloads2/Results 2nd chapter - Deltas.csv",  index_col= 11)
data = data[data.columns[2:8]].dropna()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
#data = data.T
data = data[data.columns[2:8]].dropna()
X = data
data.values[:] = X
from matplotlib import pyplot
plt.plot(data.hist())
pyplot.show()
dataa = load_iris()
X = StandardScaler().fit_transform(data[:])
feature_names = data.columns
X = MinMaxScaler().fit_transform(data[:])

#Plot covariance of features
import seaborn as sns
import numpy as np

ax = plt.axes()
f, ax = plt.subplots(figsize=(11, 9))
plt.figure()

mask = np.zeros_like(np.corrcoef(X.T), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(np.corrcoef(X.T), mask=mask,
                    square=True,
                    linewidth=.5, ax=ax,annot=True, fmt = ".2f", annot_kws={"size":25})


            ax.set_xlabel("Box Size (cm) 2020")
            ax.set_ylabel("Box Size (cm) 2019")
im = ax.imshow(np.corrcoef(X.T), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticklabels(list(feature_names), rotation=20, fontsize = 20)
ax.set_yticklabels(list(feature_names),  rotation=0, fontsize = 25)
x.set_title("3D Change detection correlation matrix",fontsize = 35)


xs = all_models_res['C3'][1].counts_list[0]
ys= all_models_res['C3'][1].lengths_list[0]
arry = []
for y,x in enumerate(xs):
    arry.append([ys[y], x])

def FDperStep (arry):
    arr2D = np.empty(shape=(9,9))
    for i,ii in enumerate(arry):
        for j, jj in enumerate(arry):
            if i == j:
                arr2D[i,j] = 0
            if i != j:
                #print (ii, jj)
                #Delta SA/Delta Alpha(smoothing, step size)
                arr2D[i,j] = float((np.log(ii[1])-np.log(jj[1]))/(np.log(ii[0])-np.log(jj[0])))
    import pandas as pd
    df = pd.DataFrame(arr2D)
    xs = [int(x[0]) for x in arry]
    df.columns = xs
    df.index = xs
    return(df)
def plotHeatMap(df,slopee, dropDuplicates = True):
        # Exclude duplicate correlations by masking uper right values
        if dropDuplicates:
            mask = np.zeros_like(df, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

        # Set background color / chart style
        sns.set_style(style = 'white')

        # Set up  matplotlib figure
        f, ax = plt.subplots(figsize=(9, 9))
         # Add diverging colormap from red to blue
        cmap = sns.diverging_palette(50, 10, as_cmap=True)
        cmap = sns.color_palette("flare")
        # Draw correlation plot with or without duplicates
        if dropDuplicates:
            sns.heatmap(df, mask=mask, cmap=cmap,
                    square=True,
                    linewidth=.5, ax=ax,annot=True, fmt = ".2f")
            ax.set_xlabel("Box Size (cm)")
            ax.set_ylabel("Box Size (cm)")
            #ax.set_title(r"$ (\delta ~log(NumBoxes))/(\delta ~log(BoxSize))$" )+"\n"
           #              r"FD breakdown, slope = %.3f" %all_models_res['C3'][1].coeffs[1])
            ax.set_title(r"$\frac {(\delta ~log(NumBoxes))} {(\delta ~log(BoxSize)}~by~box~size$")
            ax.legend()
        else:
            sns.heatmap(df, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
df = FDperStep(arry)
plotHeatMap(df, slopee, dropDuplicates = True)

##Per step differences:
#step1 generate average matrix per model
import pandas as pd

for mdl in all_models_res:
    xspyear = []
    yspyear = []
    violinArry = pd.DataFrame()
    for year = all_models_res[mdl][1]:
        print(year.year)
        xs = []
        ys = []
        avg = []

        for i,rep in enumerate(year.counts_list):
            ys.append(year.counts_list[i])
            xs.append(year.lengths_list[i])

            #print(year.counts_list[i])
        xdData = pd.DataFrame(xs)
        xData = pd.DataFrame(xs)
        yData = pd.DataFrame(ys)
        xData = np.log(xData)
        yData = np.log(yData)
        meanX, stdX = xdData.median(axis=0), xData.std(axis=0)

        violinplotdata = pd.DataFrame({'BoxSize': xData.melt().value,
                                       'SizeCat': xData.melt().variable,
                                       'NumBoxes':yData.melt().value,
                                       'SizeAve': np.repeat(round(meanX, 3), 5).values,
                                       'Year': np.repeat(year.year, len(xData.melt().value))})

        violinArry = violinArry.append([violinplotdata])

        from scipy.stats import linregress as lr
        ax = plt.figure()
        ax = sns.violinplot(x="SizeCat", y="NumBoxes", hue = "Year", data=violinplotdata, transperancy = 0.5)
        ax = sns.regplot(x="BoxSize", y="NumBoxes", data=violinplotdata , fit_reg = True)
    slope, intercept, r_value, p_value, std_err = lr(x = ax.get_lines()[0].get_xdata(),y=ax.get_lines()[0].get_ydata())
        ax.set_xticklabels(list(round(meanX, 3)), rotation=0)
        ax.set_xlabel("log(Box Size) (cm)")
        ax.set_ylabel("log(numBox)")
        ax.set_title("violin plot of box counting method")
            ax.x
        ##Workf from here
        yData
        xData.mean()
        plt.scatter(meanX, meanY)
        plt.scatter(meanX, meanY)
        meanX, stdX = xData.median(axis = 0 ) , xData.std  (axis = 0 )
        meanY, stdY = yData.median(axis = 0 ) , yData.std  (axis = 0 )
        #stdY[stdY == -np.inf] = 0
        plt.figure()
        plt.errorbar(meanX, meanY, xerr= stdX, yerr= stdY)
        plt.scatter(xData, yData)

        plt.annotate()
        for j in mdl:
        print(j)

barplotdata = pd.DataFrame(columns=['Difference', 'SizeCat','model'])
import seaborn as sns
for mdl in all_models_res:
    print (mdl)
    #mdl ='C1'
    xspyear = []
    yspyear = []
    #violinArry = pd.DataFrame()
    year = all_models_res[mdl][0]
    for i, _ in enumerate(year.counts_list):
        print(i)
        fdpstepArry = []
        stepsArry = []
        for year in all_models_res[mdl]:
            xs = year.counts_list[i]
            ys= year.lengths_list[i]
            stepsArry.append(ys)
            arry = []
            for y,x in enumerate(xs):
                arry.append([ys[y], x])
            fdpstep = FDperStep (arry)
            fdpstepArry.append(fdpstep)
        d1, d2  = pd.DataFrame(fdpstepArry[0]), pd.DataFrame(fdpstepArry[1])
        if d1.columns.size > d2.columns.size:
            d1 = d1.drop(d1.columns[d1.columns.size-1])
            d1 = d1.drop(d1.columns[d1.columns.size-1], axis=1)
        if d1.columns.size < d2.columns.size:
            d2 = d1.drop(d2.columns[d2.columns.size - 1])
            d2 = d1.drop(d2.columns[d2.columns.size - 1], axis=1)
        df = np.subtract(np.asmatrix(d1), np.asmatrix(d2))
#to make barplot from perstep difference df
        arryPstepDiff =[]
        steps =[]
        for i in range(len(df)-1):
            arryPstepDiff.append(df[i,i+1 ])
        for i in range(len(arryPstepDiff)):
            barplotdata = barplotdata.append({'BoxSize2019': stepsArry[0][i],
                                              'BoxSize2020':stepsArry[1][i],
                                             'Difference': arryPstepDiff[i],
                                             'SizeCat': i,
                                             'model':mdl}, ignore_index=True)

barplotdata = barplotdata[barplotdata['SizeCat']<8]
medians19 = barplotdata['BoxSize2019'].groupby(barplotdata['SizeCat']).median()
medians20= barplotdata['BoxSize2020'].groupby(barplotdata['SizeCat']).median()
std19 = barplotdata['BoxSize2019'].groupby(barplotdata['SizeCat']).std()
std20= barplotdata['BoxSize2020'].groupby(barplotdata['SizeCat']).std()
ax = plt.figure()
ax = sns.barplot(x="SizeCat", y="Difference", hue="model", data= barplotdata)
xtcks19  = [("%.0f +-" %medians19[i] + "%.2f" %std19[i]) for i in range(len(medians19)) ]
xtcks20  = [("%.0f +-" %medians20[i] + "%.2f" %std20[i]) for i in range(len(medians20)) ]
ax.set_xticklabels(xtcks, rotation=0)
ax.text(0.01, 0.01,
        (xtcks20 )
        ,verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='green', fontsize=25)
ax.text(0.01, 0.07,
        (xtcks19)
        ,verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='red', fontsize=25)
ax.text(0.05, 0.12,
        ("step size med +-std, 2019 top, 2020 bottom:")
        ,verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=25)
ax.set_title("difference in FD by step size")

def FDperStep (arry):
    arr2D = np.empty(shape=(len(ys),len(ys)))
    for i,ii in enumerate(arry):
        for j, jj in enumerate(arry):
            if i == j:
                arr2D[i,j] = 0
            if i != j:
                #print (ii, jj)
                #Delta SA/Delta Alpha(smoothing, step size)
                arr2D[i,j] = float((np.log(ii[1])-np.log(jj[1]))/(np.log(ii[0])-np.log(jj[0])))
    import pandas as pd
    df = pd.DataFrame(arr2D)
    xs = [int(x[0]) for x in arry]
    df.columns = xs
    df.index = xs
    return(df)
def plotHeatMap(df,slopee, dropDuplicates = True):
        # Exclude duplicate correlations by masking uper right values
        if dropDuplicates:
            mask = np.zeros_like(df, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

        # Set background color / chart style
        sns.set_style(style = 'white')

        # Set up  matplotlib figure
        f, ax = plt.subplots(figsize=(9, 9))
         # Add diverging colormap from red to blue
        cmap = sns.diverging_palette(50, 10, as_cmap=True)
        cmap = sns.color_palette("flare")
        # Draw correlation plot with or without duplicates
        if dropDuplicates:
            sns.heatmap(df, mask=mask, cmap=cmap,
                    square=True,
                    linewidth=.5, ax=ax,annot=True, fmt = ".2f", annot_kws={"size":25})
            ax.set_xticklabels([round(y, 1) for y in stepsArry[0]], rotation = 30)
            ax.set_yticklabels([round(y, 1) for y in stepsArry[1]], rotation=30)

            ax.set_xlabel("Box Size (cm) 2020")
            ax.set_ylabel("Box Size (cm) 2019")
            #ax.set_title(r"$ (\delta ~log(NumBoxes))/(\delta ~log(BoxSize))$" )+"\n"
           #              r"FD breakdown, slope = %.3f" %all_models_res['C3'][1].coeffs[1])
            ax.set_title(r"$\frac {(\delta ~log(NumBoxes))} {(\delta ~log(BoxSize)}~by~box~size$")
            ax.set_title(r"$\frac {(\delta ~log(NumBoxes))} {(\delta ~log(BoxSize)}~per~step~diff~Years~Kzaa$")

            ax.legend()
        else:
            sns.heatmap(df, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)


def myplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='green', ha='center',
                     va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


myplot(pca[:, 0:2], np.transpose(pcamodel.components_[0:2, :]), list(x.columns))
plt.show()