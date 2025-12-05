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


# ============================
#  RESUME SUPPORT FUNCTIONS
# ============================
def get_last_saved_index(output_path):
    raw_dir = os.path.join(output_path, "raw")
    if not os.path.isdir(raw_dir):
        return -1
    files = [f for f in os.listdir(raw_dir) if f.endswith("_data.png")]
    if len(files) == 0:
        return -1
    nums = [int(f.split("_")[1]) for f in files]
    return max(nums)


# ============================================================
#  ORIGINAL CODE (unchanged except where resume added)
# ============================================================

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

vector_norm = 25.0 


def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):

    imageio.imwrite(path+'raw/sample_'+str(idx).zfill(6)+'_data.png', patch)
    imageio.imwrite(path+'seg/sample_'+str(idx).zfill(6)+'_seg.png', patch_seg)

    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    mesh.lines = patch_edge.flatten()
    mesh.save(path+'vtp/sample_'+str(idx).zfill(6)+'_graph.vtp')


def patch_extract(save_path, image, seg, mesh):

    global image_id
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

        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :], ((pad_h,pad_h),(pad_w,pad_w),(0,0)))
        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w], ((pad_h,pad_h),(pad_w,pad_w)))

        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]
        clipped_mesh = mesh.clip_box(bounds, invert=False)

        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes==1)*2:].reshape(-1,3)

        # filtering indices
        patch_coord_ind = np.where((np.prod(patch_coordinates>=start, 1)*np.prod(patch_coordinates<=end, 1))>0.0)
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]
        patch_edge = [tuple(l) for l in patch_edge[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]

        temp = np.array(patch_edge).flatten()
        temp = [np.where(patch_coord_ind[0] == ind) for ind in temp]
        patch_edge = np.array(temp).reshape(-1,2)

        if patch_coordinates.shape[0] < 2 or patch_edge.shape[0] < 1:
            continue
        
        patch_coordinates = (patch_coordinates-start+np.array(pad))/np.array(patch_size)

        # SAVE ============
        if patch_seg.sum() > 10:
            save_input(save_path, image_id, patch, patch_seg, patch_coordinates, patch_edge)
            image_id += 1



# =================================================================
#  TRAIN / TEST CITY SPLIT
# =================================================================
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



# =================================================================
#                     MAIN PROGRAM (Resume enabled)
# =================================================================
if __name__ == "__main__":

    root_dir = "./data/20cities/"

    # ================================
    #  TRAIN
    # ================================
    train_path = './data/20cities/train_data/'
    os.makedirs(train_path+"/raw", exist_ok=True)
    os.makedirs(train_path+"/seg", exist_ok=True)
    os.makedirs(train_path+"/vtp", exist_ok=True)

    # FIND LAST SAVED ID ---------------
    last_id = get_last_saved_index(train_path)
    image_id = last_id + 1
    print("=== TRAIN: Continue from ID:", image_id)

    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in indrange_train:
        raw_files.append(root_dir + "/region_%d_sat" % ind)
        seg_files.append(root_dir + "/region_%d_gt.png" % ind)
        vtk_files.append(root_dir + "/region_%d_refine_gt_graph.p" % ind)
        
    print("Preparing Train Data")

    for ind in range(len(raw_files)):

        print("Train region:", ind)

        try:
            sat_img = imageio.imread(raw_files[ind]+".png")
        except:
            sat_img = imageio.imread(raw_files[ind]+".jpg")

        with open(vtk_files[ind], 'rb') as f:
            graph = pickle.load(f)

        node_array, edge_array = convert_graph(graph)
        gt_seg = imageio.imread(seg_files[ind])

        patch_coord = np.concatenate((node_array, np.int32(np.zeros((node_array.shape[0],1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate((np.int32(2*np.ones((edge_array.shape[0],1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()

        patch_extract(train_path, sat_img, gt_seg, mesh)



    # ================================
    #  TEST
    # ================================
    test_path = './data/20cities/test_data/'
    os.makedirs(test_path+"/raw", exist_ok=True)
    os.makedirs(test_path+"/seg", exist_ok=True)
    os.makedirs(test_path+"/vtp", exist_ok=True)

    last_id = get_last_saved_index(test_path)
    image_id = last_id + 1
    print("=== TEST: Continue from ID:", image_id)

    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in indrange_test:
        raw_files.append(root_dir + "/region_%d_sat" % ind)
        seg_files.append(root_dir + "/region_%d_gt.png" % ind)
        vtk_files.append(root_dir + "/region_%d_refine_gt_graph.p" % ind)

    print("Preparing Test Data")

    for ind in range(len(raw_files)):

        print("Test region:", ind)

        try:
            sat_img = imageio.imread(raw_files[ind]+".png")
        except:
            sat_img = imageio.imread(raw_files[ind]+".jpg")

        with open(vtk_files[ind], 'rb') as f:
            graph = pickle.load(f)

        node_array, edge_array = convert_graph(graph)
        gt_seg = imageio.imread(seg_files[ind])

        patch_coord = np.concatenate((node_array, np.int32(np.zeros((node_array.shape[0],1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate((np.int32(2*np.ones((edge_array.shape[0],1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()

        patch_extract(test_path, sat_img, gt_seg, mesh)
