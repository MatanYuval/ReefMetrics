import pandas as pd
mesh = o3d.io.read_triangle_mesh("C:/Users/USER/Downloads/ObsC3.ply")
random_angle_list = np.random.rand(5, 3) * 2 * np.pi
R_list = [o3d.geometry.get_rotation_matrix_from_axis_angle(random_angle) for random_angle in random_angle_list]
nm = "C4_2022.ply"
nm = str(DATA_ROOT/nm)
mesh = o3d.io.read_triangle_mesh(nm)
#("C:/Users/USER/Downloads/ObsC3.ply")
pcd = o3d.geometry.PointCloud()
##SAMPLE POINTS UNIFORMLY FROM MESH FOR MASS
pcd.points = mesh.vertices
pcd.translate(-np.asarray(pcd.points).min(0))
pcd.scale(100, center=np.array([0, 0, 0]))
coor = np.asarray(pcd.points)
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 6))
slopee20 = []
df20 = []
slop19 = []
slop20 = []

for R, _ in enumerate(R_list):
    print(R)
    pcd = pcd.rotate(R_list[R])
    pcd.translate(-np.asarray(pcd.points).min(0))
    coor = np.asarray(pcd.points)
    tree = BoxCounting.get_tree(coor)
    dff = pd.DataFrame({'Depth': [],
                        'Parent': [],
                        # 'Parentt':[],
                        'NumKids': [],
                        'NumPoints': [],
                        'BoxSize': [],
                        }, dtype="category")
    dataToPlot = lacunarity(tree, dff)
    xs, ys, coeffs, slope = get_xs_ys(dataToPlot)
    sns.regplot(x=xs, y=ys, ax=ax, label=f" Fitted line {coeffs[1]:.3f}X+{coeffs[0]:.3f}")
    slopee20.append(slope)
    df20.append([xs,ys])
#legend = []
def lacunarity (tree,df):
    if tree is None:
        return (df)
    for i, branch in enumerate(tree.children):
        if branch is None:
            continue
        if branch.children_num == 0:
            continue
        df = df.append({'NumKids': int(branch.children_num),
                    'Depth': int(branch.depth),
                    'Parent': i,
                    'NumPoints': int(branch.coor_num),
                    'BoxSize':int(list(branch.box_counting)[0]),
                    }, ignore_index=True)
        #print(df)
        df = lacunarity(branch, df)
    return (df)
def get_xs_ys(dataToPlot):
        distinct_keys = dataToPlot['BoxSize'].unique()
        df_subset = {}
        # pd.DataFrame(  columns = distinct_keys)
        for i, key in enumerate(distinct_keys):
            df_subset[key]= pd.DataFrame(dataToPlot[dataToPlot.BoxSize == key])
        #plt.hist(pd.cut(dd['NumPoints'], 10, precision=0, duplicates="drop", labels=False))
        ##From paper : Higher lacunarity values represent greater relative clumping of a habitat type or, alternatively, a widerrange of gap sizes in the distribution of the habitat.
        df_Lac = pd.DataFrame(columns=['PQ', 'P2Q', 'BoxSize', 'lnBoxSize', 'Lac', 'lac2', 'lnLac', 'lnLac2'])
        parents_perLevel = [1]
        cntr = 0
        for i in df_subset:
            parents_perLevel.append(sum(df_subset[i]['NumKids'].value_counts()))
        for i in df_subset:
            print (i)
            # Probability Distribution
            aa = {0: 0}
            a = dict(df_subset[i]['NumKids'].value_counts())
            aa.update(a)
            a = dict(sorted(aa.items()))
            depthh = df_subset[i]['Depth'].unique()[0]
            #print (a)
            # Freq. Distribution
            #Define SEARCH SPACE!**CHECK THIS
            total_theoretical_kids = 8 ** (depthh)
            #if cntr <2: total_theoretical_kids = parents_perLevel[cntr]*8
                #print("yes",cntr, parents_perLevel)
           # if cntr > 1:                total_theoretical_kids = parents_perLevel[2] * (8 ** (cntr-1))
                #print(cntr, parents_perLevel)
            cntr += 1
            print(total_theoretical_kids)
            #print(total_theoretical_kids)
            a[0] = total_theoretical_kids - sum(a.values())
            total_kids = sum(a.values())
            print(total_kids)
            mean_kids, Var_kids = np.mean(list(a.values())), np.var(list(a.values()))
            lac2 = (Var_kids / (mean_kids ** 2)) + 1  # From Paper https://link.springer.com/content/pdf/10.1007/BF00125351.pdf
            # First Moment:
            pq = sum(list(map(lambda i, j: (i / total_theoretical_kids) * j, a.values(), a.keys())))
            # Second Moment:
            p2q = sum(list(map(lambda i, j: (i / total_theoretical_kids) * (j ** 2), a.values(), a.keys())))
            lac = round(p2q / (pq ** 2), 3)
            df_Lac = df_Lac.append({  # 'P': a.keys(),
                # 'N': a.values(),
                # 'Q': [i/8 for i in a.values()],
                # 'PQ': pq,
                # 'P2Q': p2q,
                'BoxSize': i,
                'lnBoxSize': np.log(i),
                'Lac': lac,
                'lac2': lac2,
                'lnLac': np.log(lac),
                'lnLac2': np.log(lac2),
            }, ignore_index=True)
            print(df_Lac)
            # BoxSizeDf = BoxSizeDf[['BoxSize', 'NumKids']]
        ys, xs = df_Lac.lnLac, df_Lac.lnBoxSize
        coeffs, _ = np.polynomial.polynomial.polyfit(xs, ys, 1, full=True)
        slope =  coeffs[1]
        print(f"coefficient (Lacunarity) = {slope: .3f}")
        return (xs, ys, coeffs,slope)

##TEST METHOD ACCURACY
xs20 = []
ys20 = []
for i in df20:
    for cntr, j in enumerate(i):
        for k in j:
            if cntr % 2 == 0:
                xs20.append(k)
                print (cntr)
            else:
                ys20.append(k)
                print(cntr)
# plot
coeffs, _ = np.polynomial.polynomial.polyfit(xs19, ys19, 1, full=True)
slope = coeffs[1]
sns.regplot(x=xs19, y=ys19, ax=ax,   label=f" Fitted line {coeffs[1]:.3f}X+{coeffs[0]:.3f}")
plt.boxplot([slopee, slopee20])
#sns.regplot(x=xs, y=ys, ax=ax)            label=f"{slope}: Fitted line {coeffs[1]:.3f}X+{coeffs[0]:.3f}")
ax.set_title("Lacunarity of coral reef")
ax.set_ylabel("$\log Lacunarity $")
ax.set_xlabel("$\log \epsilon$")
#ax.set_title(f"Box counting fit: {model_res[0].model} (average over {len(model_res[0])} repetitions)")
ax.legend()
plt.show()


def get_xs_ys_byNUMPOINTS(dataToPlot):
    distinct_keys = dataToPlot['BoxSize'].unique()
    df_subset = {}
    for i, key in enumerate(distinct_keys):
         df_subset[key] = pd.DataFrame(dataToPlot[dataToPlot.BoxSize == key])
    #plt.hist(pd.cut(dd['NumPoints'], 10, precision=0, duplicates="drop", labels=False))
    #From paper: Higher lacunarity values represent greater relative clumping of a habitat type or, alternatively, a widerrange of gap sizes in the distribution of the habitat.
    df_Lac = pd.DataFrame(columns=['PQ', 'P2Q', 'BoxSize', 'lnBoxSize', 'Lac', 'lac2', 'lnLac', 'lnLac2'])
    for i in df_subset:
        # Probability Distribution
        dd = pd.DataFrame(df_subset[i])
        depthh = df_subset[i]['Depth'].unique()[0]
        dd['PointCats'] = pd.cut(dd['NumPoints'], 7, precision=0, duplicates="drop", labels=[i for i in range(7)])
        #total_theoretical_kids = 8 ** depthh
        a = dict(dd['PointCats'].value_counts())
        aa = {}
        for ii in range(7):
            aa[ii + 1] = a.pop(ii)
        a = aa
        total_kids = sum(a.values())
        a = {0: 0}
        a.update(aa)
        a = dict(sorted(a.items()))
        # print (a)
        # Freq. Distribution
        # Define SEARCH SPACE!**CHECK THIS
        total_theoretical_kids = 8 ** (depthh)
        # if cntr <2: total_theoretical_kids = parents_perLevel[cntr]*8
        # print("yes",cntr, parents_perLevel)
        # if cntr > 1:total_theoretical_kids = parents_perLevel[2] * (8 ** (cntr-1))
        # print(cntr, parents_perLevel)
        #cntr += 1
        print(total_theoretical_kids)
        # print(total_theoretical_kids)
        a[0] = total_theoretical_kids - sum(a.values())
        total_kids = sum(a.values())
        print(total_kids)
        mean_kids, Var_kids = np.mean(list(a.values())), np.var(list(a.values()))
        lac2 = (Var_kids / (mean_kids ** 2)) + 1  # From Paper https://link.springer.com/content/pdf/10.1007/BF00125351.pdf
        # First Moment:
        pq = sum(list(map(lambda i, j: (i / total_kids) * (j), a.values(), a.keys())))
        # Second Moment:
        p2q = sum(list(map(lambda i, j: (i / total_kids) * ((j) ** 2), a.values(), a.keys())))
        lac = round(p2q / (pq ** 2), 3)
        df_Lac = df_Lac.append({# 'P': a.keys(),
             #'N': a.values(),
             #'Q': [i/8 for i in a.values()],
             #'PQ': pq,
             #'P2Q': p2q,
            'BoxSize': i,
            'lnBoxSize': np.log(i),
            'Lac': lac,
            'lnLac': np.log(lac),
            'lac2': round(lac2, 3),
            'lnLac2': np.log(lac2),
            }, ignore_index=True)
    print (df_Lac)
    ys, xs = df_Lac.lnLac, df_Lac.lnBoxSize
    coeffs, _ = np.polynomial.polynomial.polyfit(xs, ys, 1, full=True)
    slope =  coeffs[1]
    print(f"coefficient (Lacunarity) = {slope: .3f}")
    return (xs, ys, coeffs,slope)
#plot ::
import matplotlib.pyplot as plt
import numpy as np
distinct_keys = dataToPlot['Depth'].unique()
#fig, axes = plt.subplots(len(distinct_keys),  sharex=True)
fig, axes = plt.subplots(2,3, sharex=True, sharey=False)
fig.suptitle("Lacunarity from box counting- NumBox by depth, Mean, STD ", fontsize=16)
for i, key in enumerate(distinct_keys):
    df_subset = dataToPlot[dataToPlot.Depth==key]
    df_subset= df_subset[["Depth", "NumKids"]]
    if i ==0:
        print (df_subset)

    mu = np.round(np.mean(df_subset['NumKids']), 3)  # mean of distribution
    sigma = np.round(np.std(df_subset['NumKids']), 3)  # standard deviation of distribution
    # {Ready to plot histogram data per depth}
    if i<2:
       #axes[0,0].plt.hist(df_subset['NumKids'], 8, facecolor='g', alpha=0.75)
        df_subset['NumKids'].plot(kind='hist', fill=True, log=False, ax = axes[i,0], title = str([key, mu, sigma]))
    if 1<i<4:
        #plt.hist(df_subset['NumKids'], 8, facecolor='g', alpha=0.75, ax = axes[i-2,0], title = key)
        df_subset['NumKids'].plot(kind='hist',fill=True, log=False,ax = axes[i-2,1],title =str([key, mu, sigma]))
    if 3<i :
        #plt.hist(df_subset['NumKids'], 8, facecolor='g', alpha=0.75, ax = axes[i-4,0], title = key)
        df_subset['NumKids'].plot(kind='hist', fill=True, log=False,ax = axes[i-4,2], title = str([key, mu, sigma]))

fig, ax = plt.subplots()

# the histogram of the data
#mu = np.mean(df_subset['NumKids'])  # mean of distribution
#sigma = np.std(df_subset['NumKids'])  # standard deviation of distribution
plt.show()

distinct_keys = dataToPlot['BoxSize'].unique()
#fig, axes = plt.subplots(len(distinct_keys),  sharex=True)
fig, axes = plt.subplots(2,2)
fig.suptitle("Lacunarity from box counting- NumPoints by boxSize, Mean, STD ", fontsize=16)
for i, key in enumerate(distinct_keys):
    df_subset = dataToPlot[dataToPlot.BoxSize==key]
    df_subset= df_subset[['BoxSize', 'NumPoints']]
    if i ==0:
        continue
    mu = np.round(np.mean(df_subset['NumPoints']), 3)  # mean of distribution
    sigma = np.round(np.std(df_subset['NumPoints']), 3)  # standard deviation of distribution
    # {Ready to plot histogram data per depth}
    if i<3:
       #axes[0,0].plt.hist(df_subset['NumKids'], 8, facecolor='g', alpha=0.75)
        df_subset['NumPoints'].plot(kind='hist', label="NumPoints", bins = 50, fill=True, log=False, ax = axes[i-1,0], title = str([key, mu, sigma]))
    if 2<i<6:
        #plt.hist(df_subset['NumKids'], 8, facecolor='g', alpha=0.75, ax = axes[i-2,0], title = key)
        df_subset['NumPoints'].plot(kind='hist',bins = 50, xlabel="NumPoints", fill=True, log=False,ax = axes[i-3,1],title =str([key, mu, sigma]))
    if 3<i :
        #plt.hist(df_subset['NumKids'], 8, facecolor='g', alpha=0.75, ax = axes[i-4,0], title = key)
        df_subset['NumPoints'].plot(kind='hist',bins = 50, fill=True, log=False,ax = axes[i-5,2], title = str([key, mu, sigma]))

fig, ax = plt.subplots()

# dataToPlot['Depth']= pd.Categorical(dataToPlot['Depth'])
# Frequency distribution per boxSize and box counts (not points)

# legend += [f"{}: Fitted line {coeffs[1]:.3f}X+{coeffs[0]:.3f}",
#          f"{}: Measured points"]
# FOr testing: tree1=tree1.children[3].children[2]




slope199 = []
slope200 = []
for i in all_models_res:
    for j in i.counts_list, k in i.lengths_list:
        print (k)
        for k in j:
            xs = j.c
    for cntr, j in enumerate(i):
        for k in j:
            if cntr % 2 == 0:
                xs20.append(k)
                print (cntr)
            else:
                ys20.append(k)
                print(cntr)
