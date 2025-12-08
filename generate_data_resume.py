import pdb
import math
import imageio
import pyvista
import numpy as np
import pickle
import random
import os

patch_size = [128,128,1]
pad = [5,5,0]

# ====================================================
# RESUME SUPPORT
# ====================================================
def get_last_saved_index(path):
    raw_dir = os.path.join(path, "raw")
    if not os.path.isdir(raw_dir):
        return -1
    files = [f for f in os.listdir(raw_dir) if f.endswith("_data.png")]
    if len(files) == 0:
        return -1
    nums = [int(f.split("_")[1]) for f in files]
    return max(nums)

# ====================================================
# BASE FUNCTIONS (UNMODIFIED)
# ====================================================
def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min = -1, a_max=1))

def convert_graph(graph):
    node_list = []
    edge_list = []
    for n, v in graph.items():
        node_list.append(n)
    node_array = np.array(node_list)

    for ind, (n, v) in enumerate(graph.items()):
        for nei in v:
            idx = node_list.index(nei)
            edge_list.append(np.array((ind,idx)))
    edge_array = np.array(edge_list)
    return node_array, edge_array

def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):
    imageio.imwrite(path + 'raw/sample_' + str(idx).zfill(6) + '_data.png', patch)
    imageio.imwrite(path + 'seg/sample_' + str(idx).zfill(6) + '_seg.png', patch_seg)

    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    mesh.lines = patch_edge.flatten()
    mesh.save(path + 'vtp/sample_' + str(idx).zfill(6) + '_graph.vtp')

def prune_patch(patch_coord_list, patch_edge_list):
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:,0], edge[:,1]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)
        dist_adj[edge[:,1], edge[:,0]] = dist_adj[edge[:,0], edge[:,1]]

        start = True
        node_mask = np.ones(coord.shape[0], dtype=np.bool)
        while start:
            degree = (dist_adj>0).sum(1)
            deg_2 = list(np.where(degree==2)[0])
            if len(deg_2)==0:
                start = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx,:]>0)[0]

                p1 = coord[idx,:]
                p2 = coord[deg_2_neighbor[0],:]
                p3 = coord[deg_2_neighbor[1],:]
                l1 = p2-p1
                l2 = p3-p1
                node_angle = angle(l1,l2)*180/math.pi
                if node_angle>160:
                    node_mask[idx]=False
                    dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]] = np.sum((p2-p3)**2)
                    dist_adj[deg_2_neighbor[1], deg_2_neighbor[0]] = np.sum((p2-p3)**2)
                    dist_adj[idx,:] = 0
                    dist_adj[:,idx] = 0
                    break
                elif n==len(deg_2)-1:
                    start = False

        new_coord = coord[node_mask,:]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj)>0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)

    return mod_patch_coord_list, mod_patch_edge_list

def patch_extract(save_path, image, seg, mesh, start_id):

    image_id = start_id
    p_h, p_w, _ = patch_size
    pad_h, pad_w, _ = pad
    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    
    h, w, d= image.shape
    x_ = np.int32(np.linspace(5, h-5-p_h, 32))
    y_ = np.int32(np.linspace(5, w-5-p_w, 32))
    
    ind = np.meshgrid(x_, y_, indexing='ij')

    for i, start in enumerate(list(np.array(ind).reshape(2,-1).T)):
        start = np.array((start[0],start[1],0))
        end = start + np.array(patch_size)-1 -2*np.array(pad)

        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :], 
                       ((pad_h,pad_h),(pad_w,pad_w),(0,0)))

        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w], 
                           ((pad_h,pad_h),(pad_w,pad_w)))

        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]

        clipped_mesh = mesh.clip_box(bounds, invert=False)
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes==1)*2:].reshape(-1,3)

        patch_coord_ind = np.where((np.prod(patch_coordinates>=start, 1)*np.prod(patch_coordinates<=end, 1))>0.0)
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]
        patch_edge = [tuple(l) for l in patch_edge[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]

        temp = np.array(patch_edge).flatten()
        temp = [np.where(patch_coord_ind[0] == ind) for ind in temp]
        patch_edge = np.array(temp).reshape(-1,2)

        if patch_coordinates.shape[0]<2 or patch_edge.shape[0]<1:
            continue

        patch_coordinates = (patch_coordinates-start+np.array(pad))/np.array(patch_size)
        mod_coord, mod_edge = prune_patch([patch_coordinates],[patch_edge])

        if patch_seg.sum()>10:
            save_input(save_path, image_id, patch, patch_seg, mod_coord[0], mod_edge[0])
            print(f"[DONE] sample {image_id} created.")
            image_id += 1

    return image_id


# ====================================================
# TRAIN / TEST SPLIT
# ====================================================
indrange_train = []
indrange_test = []

for x in range(180):
    if x % 10 < 8 :
        indrange_train.append(x)
    if x % 10 == 9:
        indrange_test.append(x)
    if x % 20 == 18:
        indrange_train.append(x)
    if x % 20 == 8:
        indrange_test.append(x)

# ====================================================
# MAIN (RESUMABLE)
# ====================================================
if __name__ == "__main__":
    
    root_dir = "/content/drive/MyDrive/relformer_data/20cities/"

    # ============= TRAIN =============
    train_path = root_dir + "train_data/"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(train_path+"raw", exist_ok=True)
    os.makedirs(train_path+"seg", exist_ok=True)
    os.makedirs(train_path+"vtp", exist_ok=True)

    last_id = get_last_saved_index(train_path)
    print("Resume Train Data from ID =", last_id+1)
    image_id = last_id + 1

    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in indrange_train:
        raw_files.append(root_dir + "region_%d_sat" % ind)
        seg_files.append(root_dir + "region_%d_gt.png" % ind)
        vtk_files.append(root_dir + "region_%d_refine_gt_graph.p" % ind)

    for ind in range(len(raw_files)):
        try:
            sat_img = imageio.imread(raw_files[ind] + ".png")
        except:
            sat_img = imageio.imread(raw_files[ind] + ".jpg")

        with open(vtk_files[ind], 'rb') as f:
            graph = pickle.load(f)
        node_array, edge_array = convert_graph(graph)

        seg = imageio.imread(seg_files[ind])

        patch_coord = np.concatenate((node_array, np.zeros((node_array.shape[0],1))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate((np.ones((edge_array.shape[0],1))*2, edge_array), 1)
        mesh.lines = patch_edge.flatten()

        image_id = patch_extract(train_path, sat_img, seg, mesh, image_id)

    # ============= TEST =============
    test_path = root_dir + "test_data/"
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(test_path+"raw", exist_ok=True)
    os.makedirs(test_path+"seg", exist_ok=True)
    os.makedirs(test_path+"vtp", exist_ok=True)

    last_id = get_last_saved_index(test_path)
    print("Resume Test Data from ID =", last_id+1)
    image_id = last_id + 1

    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in indrange_test:
        raw_files.append(root_dir + "region_%d_sat" % ind)
        seg_files.append(root_dir + "region_%d_gt.png" % ind)
        vtk_files.append(root_dir + "region_%d_refine_gt_graph.p" % ind)

    for ind in range(len(raw_files)):
        try:
            sat_img = imageio.imread(raw_files[ind] + ".png")
        except:
            sat_img = imageio.imread(raw_files[ind] + ".jpg")

        with open(vtk_files[ind], 'rb') as f:
            graph = pickle.load(f)
        node_array, edge_array = convert_graph(graph)

        seg = imageio.imread(seg_files[ind])

        patch_coord = np.concatenate((node_array, np.zeros((node_array.shape[0],1))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate((np.ones((edge_array.shape[0],1))*2, edge_array), 1)
        mesh.lines = patch_edge.flatten()

        image_id = patch_extract(test_path, sat_img, seg, mesh, image_id)

