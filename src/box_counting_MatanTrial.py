import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import open3d as o3d
from sys import setrecursionlimit
import matplotlib._color_data as mcd
from collections import Counter

# setrecursionlimit(10)
from src.visual import plot_box_counting_fit

pcd_path = 'D:\\Naama\\Documents\\Projects\\BoxCounting\data\\2020100pt.pcd'
pcd = o3d.io.read_point_cloud(pcd_path)
pc_max_size = 1000
#
# trials = 5
# for t in range(trials):
#     random_angle = np.random.rand(3) * 2 * np.pi
#     R = o3d.geometry.get_rotation_matrix_from_axis_angle(random_angle)
#     pcd = pcd.rotate(R)
#     o3d.visualization.draw_geometries([pcd])

pcd.translate(-np.asarray(pcd.coor_num).min(0))
pcd.scale(pc_max_size / np.asarray(pcd.coor_num).max(), center=np.array([0, 0, 0]))
coor = np.array(pcd.coor_num)

# bbox = pcd.get_oriented_bounding_box()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

bbox = pcd.get_axis_aligned_bounding_box()
bboxpts = np.asarray(bbox.get_box_points())
depth = 1  # Number of iterations in recursion-this is stepsize
results = []  # bcounting(bbox, pcd, depth, results)


def bcounting(bbox, coor, depth, results):
    colors = [name for name in mcd.CSS4_COLORS]
    bboxpts = np.asarray(
        bbox.get_box_points())  # ind = np.lexsort(bboxpts[:,0],bboxpts[:,2])#bboxpts = bboxpts[ind]#Divide low coor_num
    cntr = bbox.get_center()
    low = bboxpts[:, 2] < cntr[2]
    high = bboxpts[:, 2] > cntr[2]
    east = bboxpts[:, 0] > cntr[0]
    west = bboxpts[:, 0] < cntr[0]
    north = bboxpts[:, 1] > cntr[1]
    south = bboxpts[:, 1] < cntr[1]
    checkss = low, high, east, west, north, south
    allcrnrs = []
    ddepth = depth
    for i in checkss:
        lowPoints = bboxpts[i]
        cntrLow = [np.average(lowPoints[:, 0]), np.average(lowPoints[:, 1]), np.average(lowPoints[:, 2])]
        midPoints = []
        for pt in lowPoints:
            for ptt in lowPoints:
                start = pt
                # print(start)
                end = ptt  # if (((len(set(start)-set(end))) > 0) & (any(end) - any(start) <  .0001)):            #break
                middle = (start + end) / 2
                midPoints.append(middle)
        midPointss = np.unique(midPoints, axis=0)
        # ax.scatter3D(midPointss[:, 0], midPointss[:, 1], midPointss[:, 2])
        allcrnrs.append(midPointss)  # print(set(start)-set(end))            #ax.scatter3D(allcrnrs[1][0], allcrnrs[:, 1], allcrnrs[:, 2])
    flat_list = [item for sublist in allcrnrs for item in sublist]
    allpts = (np.unique(flat_list, axis=0))

    # Find boxes and define subBoxes:
    #                      West-East                   South-North                 Low-High
    box_SWL_cr = np.where((allpts[:, 0] <= cntr[0]) & (allpts[:, 1] <= cntr[1]) & (allpts[:, 2] <= cntr[2]))
    box_SWH_cr = np.where((allpts[:, 0] <= cntr[0]) & (allpts[:, 1] <= cntr[1]) & (allpts[:, 2] >= cntr[2]))
    box_SEL_cr = np.where((allpts[:, 0] >= cntr[0]) & (allpts[:, 1] <= cntr[1]) & (allpts[:, 2] <= cntr[2]))
    box_SEH_cr = np.where((allpts[:, 0] >= cntr[0]) & (allpts[:, 1] <= cntr[1]) & (allpts[:, 2] >= cntr[2]))
    box_NWL_cr = np.where((allpts[:, 0] <= cntr[0]) & (allpts[:, 1] >= cntr[1]) & (allpts[:, 2] <= cntr[2]))
    box_NWH_cr = np.where((allpts[:, 0] <= cntr[0]) & (allpts[:, 1] >= cntr[1]) & (allpts[:, 2] >= cntr[2]))
    box_NEL_cr = np.where((allpts[:, 0] >= cntr[0]) & (allpts[:, 1] >= cntr[1]) & (allpts[:, 2] <= cntr[2]))
    box_NEH_cr = np.where((allpts[:, 0] >= cntr[0]) & (allpts[:, 1] >= cntr[1]) & (allpts[:, 2] >= cntr[2]))
    subBoxes = box_SWL_cr, box_SWH_cr, box_SEL_cr, box_SEH_cr, box_NWL_cr, box_NWH_cr, box_NEL_cr, box_NEH_cr
    recurse = []
    vispts = []
    for subBox in subBoxes:
        box = list()
        # print("subBox:", subBox)
        box = list(allpts[subBox])
        box.append(list(cntr))
        box3d = o3d.geometry.PointCloud()
        box3d.points = o3d.utility.Vector3dVector(box)
        bbox3d = box3d.get_axis_aligned_bounding_box()
        center = bbox3d.get_center()
        max = bbox3d.get_max_bound()
        min = bbox3d.get_min_bound()  ### check each box in the subdivision, and save the results
        flag = 0

        points_in_box = np.bitwise_and(coor < max, coor > min).all(1)
        if points_in_box.sum() > 0:
            recurse.append((bbox3d, points_in_box))
            vispts.append(np.asarray(box3d.points))

    for i, (subBox, sub_points_in_box) in enumerate(recurse):
        if depth < 5:
            print(f"{'     ' * (depth - 1)}depth {depth} box {i}")
            # print("start recursion block f 'subBox'", i, subBox)  # save box size (iteration) and counter
            color = colors[np.random.choice(range(1, 100))]
            a = vispts[i]
            ax.scatter3D(a[:, 0], a[:, 1], a[:, 2], c=color)
            results.append(depth)
            bcounting(subBox, coor[sub_points_in_box], depth+1, results)
        else:
            return


if __name__ == '__main__':
    bcounting(bbox, coor, depth, results)
    plt.show()

    depth_counter  = Counter(results)
    depths = np.array(list(depth_counter.keys()))
    lengths = 2. ** -depths
    counts = np.array(list(depth_counter.values()))
    plot_box_counting_fit(counts, lengths)
    plt.show()
