import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from src.box_counting import BoxCounting
from src.io_utils import read_model, get_models_list
from src.local_setting import DATA_ROOT
from src.visual import add_3d_axes, plot_trial, plot_box_counting_results_aggregate, BoxCountingResults
import pandas as pd

params = {'font.size': 18,
        'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'large',
        'ytick.labelsize':'large'}
plt.rcParams.update(params)

def entrophy_from_box_count_script(data_root, pcds, pc_max_size=32, trials=5):
    for pcd_name in pcds:
        print("==========================================")
        print(f"=============={pcd_name}================")
        print("==========================================")

        # load point cloud (open3d) and normalize it
        #pcd_name ="C2_19.pcd"
        #data_root=DATA_ROOT
        pcd_path = data_root / pcd_name
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        fig = plt.figure(figsize=(2 * trials, 2))
        for t in range(trials):
            ax = add_3d_axes(fig, 1, trials, t + 1)
            random_angle = np.random.rand(3) * 2 * np.pi
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(random_angle)
            # pcd = pcd.rotate(R) # TODO

            pcd.translate(-np.asarray(pcd.coor_num).min(0))
            pcd.scale(pc_max_size / np.asarray(pcd.coor_num).max(), center=np.array([0, 0, 0]))
            coor = np.asarray(pcd.coor_num)

            plot_trial(ax, coor[::100], pc_max_size)

            tree, max_power = BoxCounting.get_tree(coor)
            tree = BoxCounting.get_tree(coor)
            tree.statistic_summary_of_node(8)


            ##CHanged by Matan:
            pcd_name = "C12020.ply"
            data_root = DATA_ROOT
            pcd_path = data_root / pcd_name
            pcd = o3d.io.read_point_cloud(str(pcd_path))

            pcd.translate(-np.asarray(pcd.points).min(0))
            pcd.scale(100, center=np.array([0, 0, 0]))
            coor = np.asarray(pcd.points)

            plot_trial(ax, coor[::100], pc_max_size)

            tree = BoxCounting.get_tree(coor)
            tree.statistic_summary_of_node(max_power)


def box_counting_script(models_names_list, model_type, trials=1):
    models = get_models_list(models_list, model_type)

    all_models_res = {}
    for model, paths in models.items():
        # This list will summarize the result for this model
        model_res = []

        # creating the rotation list to aplly same rotations for both years
        random_angle_list = np.random.rand(trials, 3) * 2 * np.pi
        R_list = [o3d.geometry.get_rotation_matrix_from_axis_angle(random_angle) for random_angle in random_angle_list]

        for year, pcd_name in paths.items():
            # load point cloud (open3d) and normalize it
            pcd = read_model(pcd_name, model_type)

            print("================================================================")
            print(f"============== {f'{pcd_name} ({len(pcd.points):,} points)':<30} ================")
            print("================================================================")

            # convert to cm
            pcd.scale(100, center=np.array([0, 0, 0]))

            # This object will summarize the result for this model at this year for all trials (rotations)
            box_counting_result = BoxCountingResults(model, year, DATA_ROOT/pcd_name)

            # fig for showing the rotations of the original pcd
            # pcd_fig = plt.figure(figsize=(2 * trials, 2))
            for t, R in enumerate(R_list):
                pcd = pcd.rotate(R)
                pcd.translate(-np.asarray(pcd.points).min(0))
                coor = np.asarray(pcd.points)

                # Show current rotation
                # plot_trial(pcd_fig, trials, t, coor[::1000], )

                # perform box counting
                start = time.perf_counter()
                box_counting_tree = BoxCounting.get_tree(coor)
                end = time.perf_counter()
                sorted_box_counting_dict = box_counting_tree.get_sorted_box_counting_dict()

                # print result of current trial
                printing_dict = {'trial': f"{t}"}
                printing_dict.update(sorted_box_counting_dict)
                printing_dict['duration'] = f"{end - start:.1f} sec"
                fmtstring = (' | '.join(['{{:{:d}}}'] * len(printing_dict))).format(*[10]*len(printing_dict))
                print(fmtstring.format(*printing_dict.keys()))
                print(fmtstring.format(*printing_dict.values()))
                print(fmtstring.format(*['-'*10 for _ in [10]*len(printing_dict)]))

                box_counting_result.append(sorted_box_counting_dict)

            # after finishing all trials, calc the slope and save result
            box_counting_result.calc_box_counting_slope()
            model_res.append(box_counting_result)

        plot_box_counting_results_aggregate(model_res)
        all_models_res[model] = model_res
    return all_models_res



if __name__ == '__main__':
    models_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'NR1', 'Kzaa']
    models_list = ['IUI15C1', 'KsskyC1', 'KsskyC2','KsskyC3', 'KzaC4','NRIgloo1','NRIgloo2','NRIgloo3','NRIgloo4','NrObsC3','NrObsC4']
    models_list = ['IUI15C1','KsskyC1','NrObsC4','KzaC4']
    models_list = ['C3']
    model_type = 'mesh'
    models_list = ['BoxFull']

    all_models_res = box_counting_script(models_names_list=models_list, model_type=model_type, trials=30)
    print("")


import seaborn as sns

arry = []

legend = []
fig, ax = plt.subplots(figsize=(8, 6))
for model_res in all_models_res:
    for box_counting_res in all_models_res[model_res]:
        xs, ys = box_counting_res.get_log_values()
        slope  = box_counting_res.box_counting_slope
        coeffs = box_counting_res.coeffs
        print(f"coefficient (fractal dimension) = {slope}")
        legend += [f"{box_counting_res.model}"+f"{box_counting_res.year}: Fitted line {coeffs[1]:.3f}X+{coeffs[0]:.3f}"]
        sns.regplot(x=xs, y=ys, ax=ax,
                    label=f"{box_counting_res.model}"+f"{box_counting_res.year}: Fitted line {coeffs[1]:.3f}X+{coeffs[0]:.3f}")

    ax.set_ylabel("$\log N(\epsilon)$")
    ax.set_xlabel("$\log 1/ \epsilon$")
    #ax.set_title(f"Box counting fit: {model_res[0].model} (fit over {len(model_res[0])} repetitions)")
    ax.legend()
    plt.show()

def get_counts(mdl):
    xs = mdl.counts_list[0]
    ys= mdl.lengths_list[0]
    arry = []
    for y,x in enumerate(xs):
        arry.append([ys[y], x])
    return arry
def FDperStep (arry):
    arr2D = np.empty(shape=(len(arry),len(arry)))
    for i,ii in enumerate(arry):
        for j, jj in enumerate(arry):
            if i == j:
                arr2D[i,j] = 0
            if i != j:
                #print (ii, jj)
                #Delta SA/Delta Alpha(smoothing, step size)
                arr2D[i,j] = float((np.log(ii[1])-np.log(jj[1]))/(np.log(ii[0])-np.log(jj[0])))
    df = pd.DataFrame(arr2D)
    xs = [int(x[0]) for x in arry]
    df.columns = xs
    df.index = xs
    return(df)
def plotHeatMap(df, dropDuplicates = True):
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
#df = FDperStep(arry)
def get_diags(df, year):
    barplotdata = pd.DataFrame(columns=['Dimension', 'BoxSize','SizeCat', 'Year', 'model'])
    diagonal = pd.DataFrame([df.values[i, i+1] for i in range(len(df)-1)])
    diagonal = diagonal*-1
    for i in diagonal.index:
        a = df.index[i]
        barplotdata = barplotdata.append({'Dimension':float(diagonal.values[i]),
                                      'BoxSize': float(a),
                                      'SizeCat': i,
                                        'Year':year,
                                      'model': mdl},
                                         ignore_index=True)

    return (barplotdata)

barplotdata = pd.DataFrame(columns=['Dimension', 'BoxSize', 'SizeCat', 'Year', 'model'])
deltas = pd.DataFrame(columns=['Delta', 'SizeCat', 'Year', 'model'])
for mdl in all_models_res:
     for i in all_models_res[mdl]:
        arry = get_counts(i)
        df = FDperStep(arry)
        year = i.year
        barplotdata = barplotdata.append(get_diags(df, year))
barplotdata = barplotdata[barplotdata['SizeCat']<8]
Bsizestats = [i for i in zip(set(barplotdata['SizeCat']),
                             barplotdata['BoxSize'].groupby(barplotdata['SizeCat']).median(),
                             barplotdata['BoxSize'].groupby(barplotdata['SizeCat']).std())]
#barplotdata['Size(cm)'] =  [int(Bsizestats[barplotdata['SizeCat'].values[i]][1]) for i in range(len(barplotdata))]
deltas = pd.DataFrame(columns=['Delta FD', 'SizeCat', 'Year', 'model','Size(cm)','SizeCatStd'])
for i in range(len(barplotdata)):
    print(i)
    if barplotdata['Year'].values[i] == 2020:
        a = barplotdata[barplotdata['model'].values == barplotdata['model'].values[i]]
        a = a[a['Year'].values == 2019]
        a = a[a['SizeCat'].values == barplotdata['SizeCat'].values[i]]
        if len(a)>0:
            deltas = deltas.append({'Delta FD':(float(barplotdata['Dimension'].values[i]
                                                    - a['Dimension'].values)),
                                      'SizeCat': barplotdata['SizeCat'].values[i],
                                        'Year':2020,
                                      'model': barplotdata['model'].values[i],
                                'Size(cm)': int(Bsizestats[barplotdata['SizeCat'].values[i]][1]),
                                'SizeCatStd': Bsizestats[barplotdata['SizeCat'].values[i]][2]},
                                         ignore_index=True)
    if barplotdata['Year'].values[i] == 2022:
        a = barplotdata[barplotdata['model'].values == barplotdata['model'].values[i]]
        a = a[a['Year'].values == 2019]
        a = a[a['SizeCat'].values == barplotdata['SizeCat'].values[i]]
        if len(a) > 0:
            deltas = deltas.append({'Delta FD':(float(barplotdata['Dimension'].values[i]
                                                            -  a['Dimension'].values)),
                                    'SizeCat': barplotdata['SizeCat'].values[i],
                                    'Year': 2022,
                                    'model': barplotdata['model'].values[i],
                                    'Size(cm)': int(Bsizestats[barplotdata['SizeCat'].values[i]][1]),
                                    'SizeCatStd': Bsizestats[barplotdata['SizeCat'].values[i]][2]},
                                   ignore_index=True)
import seaborn as sns
g = sns.catplot (x="Size(cm)", y="Delta FD", hue="Year", col = "model", col_wrap=4,
                     kind="bar", data=deltas, legend = False, sharey = False)
g.set_xticklabels(rotation=90)
deltas = deltas[deltas['SizeCat']<6]
g = sns.catplot (x="model", y="Delta FD", hue="Year", col = "Size(cm)", col_wrap=3,
                     kind="bar", data=deltas, legend = False, sharey = False)
deltas = deltas[deltas['model'].values !='Kzaa']
deltas = deltas[deltas['model'].values !='NR1']
g = sns.catplot (x="Size(cm)", y="Delta FD", hue="Year",
                     kind="box", data=deltas, legend = False)
g.set(title='Change in FD from box-counting in Princess beach')
g = sns.catplot (x="Year", y="Delta FD", hue="model", col = "Size(cm)", col_wrap=4,
                     kind="bar", data=deltas, legend = False,sharey = False)
g = sns.catplot (x="Size(cm)", y="Delta FD", hue="Year", col = "model", col_wrap=4,
                     kind="bar", data=deltas, legend = False)
g = sns.catplot(x="SizeCat", y="Delta", hue="Year",
                     kind="violin", split = True, data=deltas)
g = sns.boxplot(x="SizeCat", y="Delta FD", hue="Year", data=deltas)
g = sns.swarmplot(x="SizeCat", y="Delta FD", hue="Year", data=deltas)
g = sns.catplot(x="BoxSize", y="Dimension", hue="Year",
                     kind="box", data=barplotdata)

labels = [i for i in zip(set(deltas['SizeCat']),deltas['Delta FD'].groupby(deltas['SizeCat']).median())]
# iterate through the axes containers
medians = deltas.groupby(['SizeCat','Year'])['Delta FD'].median()
medians = [np.round(i,3) for i in medians]
vertical_offset = [i +0.05 for i in medians] # offset from median for display
g = sns.boxplot(x="Size(cm)", y="Delta FD", hue="Year", data=deltas)
for i, xtick in enumerate(g.get_xticklabels()):
    print(xtick)
    if i%2 != 0:
        g.text(i-0.2, labels[i] + vertical_offset[i], str(medians[i]),
                horizontalalignment='center',size='large',color='b',weight='semibold')
    else:
        g.text(i+.2,  labels[i][1]  +float(vertical_offset[i]), str(medians[i]),
               horizontalalignment='center', size='large', color='b', weight='semibold')
data = pd.read_csv("E:/Downloads2/Results 2nd chapter - Deltas.csv")
data = pd.DataFrame(data[data.columns[:12]].dropna())
g = sns.catplot(x="BoxSize", y="Dimension", hue="Year",
                     kind="box", data=barplotdata)
data.plot(kind="box", sharey = False)
barplotdata.to_csv('E:/Downloads2/FinalExperiment/barpltdata.csv', index= False)
data = pd.DataFrame(pd.read_csv("E:/Downloads2/Results 2nd chapter - Sheet2.csv", index_col= 8))
data = pd.DataFrame(data[2:8])
data.plot(
    kind='scatter',
    subplots=True,
    sharey=False,
    figsize=(10, 6)
)
plt.subplots_adjust(wspace=0.5)

data1 = pd.melt(data, id_vars =['Modelyear','year','Model'])
data1 = data1[data1['variable']!= "Model"]
data1 = data1[data1['variable']!= "year"]
data.boxplot()
g = sns.catplot(x="variable", y="value", hue="year",
                     kind="box", data=data1, sharey = False)


    xtickk = xtick - 0.2
    g.text(xtick, 0.2, medians[xtick],
           horizontalalignment='center', size='large', color='b', weight='semibold')

fig.tight_layout()
plt.suptitle( "difference in FD by model by year")
plt.show()
xtcks = [("%.0f +-" %Bsizestats[i][1] + "%.2f" %Bsizestats[i][2]) for i in range(len(Bsizestats)-1) ]
g.set_xticklabels(xtcks, rotation=30)
g.set_xlabels("Box Size (cm) 2020")
    #for i in range(len(barplotdata.columns)):
        #plt.bar(barplotdata.index, barplotdata.values[:, i] * -1, label=barplotdata.columns[i], alpha=0.7)
    ax.legend_.remove()  # remove the individual plot legends
    ax.set_title(mdl)


plotHeatMap(df, dropDuplicates = True)
import seaborn as sns
##Per step differences:
#step1 generate average matrix per model
import pandas as pd

for mdl in all_models_res:
    xspyear = []
    yspyear = []
    violinArry = pd.DataFrame()
    for year = all_models_res[mdl][0]:
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
                                       #'SizeAve': np.repeat(round(meanX, 3), 5).values,
                                       'Year': np.repeat(year.year, len(xData.melt().value))})

        violinArry = violinArry.append([violinplotdata])

        import seaborn as sns
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
ax = sns.boxplot(x="model", y="Difference", hue="SizeCat", data= barplotdata)
ax.legend( loc='upper left', bbox_to_anchor=(1, 0.5))
ax = sns.boxplot(x="SizeCat", y="Difference", hue="model", data= barplotdata)
ax = sns.swarmplot(x="SizeCat", y="Difference" , hue="model", data= barplotdata, color=".2")
g = sns.catplot(x="SizeCat", y="Difference", hue="model", data= barplotdata, kind="swarm",
                height=4, aspect=1.5)
xtcks19  = [("%.0f +-" %medians19[i] + "%.2f" %std19[i]) for i in range(len(medians19)) ]
xtcks20  = [("%.0f +-" %medians20[i] + "%.2f" %std20[i]) for i in range(len(medians20)) ]
ax.set_xticklabels(xtcks20, rotation=0)
ax.text(0.01, 0.01,
        (xtcks20 )
        ,verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='green', fontsize=25)
ax.text(0.01, 0.01,
        (xtcks19)
        ,verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='red', fontsize=25)
ax.text(0.05, 0.07,
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

