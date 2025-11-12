import pdb
import math
import imageio
import pyvista
import numpy as np
import pickle
import random
import os
import json

checkpoint_path = "/content/drive/MyDrive/relformer_data/checkpoint.json"

# اگر فایل چک‌پوینت وجود دارد، آن را بخوان
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)
    phase = checkpoint.get("phase", "train")
    start_train = checkpoint.get("train_index", 0)
    start_test = checkpoint.get("test_index", 0)
else:
    checkpoint = {"phase": "train", "train_index": 0, "test_index": 0}
    phase = "train"
    start_train = 0
    start_test = 0


patch_size = [128,128,1]
pad = [5,5,0]

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

def neighbor_transpos(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (k[1], k[0])
		nv = []

		for _v in v :
			nv.append((_v[1],_v[0]))

		n_out[nk] = nv 

	return n_out 

def neighbor_to_integer(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (int(k[0]), int(k[1]))
		
		if nk in n_out:
			nv = n_out[nk]
		else:
			nv = []

		for _v in v :
			new_n_k = (int(_v[0]),int(_v[1]))

			if new_n_k in nv:
				pass
			else:
				nv.append(new_n_k)

		n_out[nk] = nv 

	return n_out


def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    imageio.imwrite(path+'raw/sample_'+str(idx).zfill(6)+'_data.png', patch)
    imageio.imwrite(path+'seg/sample_'+str(idx).zfill(6)+'_seg.png', patch_seg)
    
    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)
    
    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'mesh/sample_'+str(idx).zfill(4)+'_segmentation.stl')
    
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    mesh.lines = patch_edge.flatten()
    mesh.save(path+'vtp/sample_'+str(idx).zfill(6)+'_graph.vtp')


def patch_extract(save_path,image, seg,  mesh, device=None):
    """[summary]

    Args:
        image ([type]): [description]
        coordinates ([type]): [description]
        lines ([type]): [description]
        patch_size (tuple, optional): [description]. Defaults to (64,64,64).
        num_patch (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    global image_id
    p_h, p_w, _ = patch_size
    pad_h, pad_w, _ = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    
    h, w, d= image.shape
    x_ = np.int32(np.linspace(5, h-5-p_h, 32))
    y_ = np.int32(np.linspace(5, w-5-p_w, 32))
    
    ind = np.meshgrid(x_, y_, indexing='ij')
    # Center Crop based on foreground

    for i, start in enumerate(list(np.array(ind).reshape(2,-1).T)):
        # print(image.shape, seg.shape)
        start = np.array((start[0],start[1],0))
        end = start + np.array(patch_size)-1 -2*np.array(pad)

        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :], ((pad_h,pad_h),(pad_w,pad_w),(0,0)))
        patch_list = [patch]

        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w,], ((pad_h,pad_h),(pad_w,pad_w)))
        seg_list = [patch_seg]

        # collect all the nodes
        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]

        clipped_mesh = mesh.clip_box(bounds, invert=False)
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes==1)*2:].reshape(-1,3)
        
        patch_coord_ind = np.where((np.prod(patch_coordinates>=start, 1)*np.prod(patch_coordinates<=end, 1))>0.0)
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]  # all coordinates inside the patch
        patch_edge = [tuple(l) for l in patch_edge[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]
        
        temp = np.array(patch_edge).flatten()  # flatten all the indices of the edges which completely lie inside patch
        temp = [np.where(patch_coord_ind[0] == ind) for ind in temp]  # remap the edge indices according to the new order
        patch_edge = np.array(temp).reshape(-1,2)  # reshape the edge list into previous format

        if patch_coordinates.shape[0]<2 or patch_edge.shape[0]<1:
            continue
        # concatenate final variables
        patch_coordinates = (patch_coordinates-start+np.array(pad))/np.array(patch_size)
        patch_coord_list = [patch_coordinates]#.to(device))
        patch_edge_list = [patch_edge]#.to(device))

        mod_patch_coord_list, mod_patch_edge_list = prune_patch(patch_coord_list, patch_edge_list)

        # save data
        for patch, patch_seg, patch_coord, patch_edge in zip(patch_list, seg_list, mod_patch_coord_list, mod_patch_edge_list):
            if patch_seg.sum()>10:
                save_input(save_path, image_id, patch, patch_seg, patch_coord, patch_edge)
                image_id = image_id+1
                # print('Image No', image_id)

def prune_patch(patch_coord_list, patch_edge_list):
    """[summary]

    Args:
        patch_list ([type]): [description]
        patch_coord_list ([type]): [description]
        patch_edge_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):

        # find largest graph segment in graph and in skeleton and see if they match
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:,0], edge[:,1]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)
        dist_adj[edge[:,1], edge[:,0]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)

        # straighten the graph by removing redundant nodes
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
                node_angle = angle(l1,l2)*180 / math.pi
                if node_angle>160:
                    node_mask[idx]=False
                    dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]] = np.sum((p2-p3)**2)
                    dist_adj[deg_2_neighbor[1], deg_2_neighbor[0]] = np.sum((p2-p3)**2)

                    dist_adj[idx, deg_2_neighbor[0]] = 0.0
                    dist_adj[deg_2_neighbor[0], idx] = 0.0
                    dist_adj[idx, deg_2_neighbor[1]] = 0.0
                    dist_adj[deg_2_neighbor[1], idx] = 0.0
                    break
                elif n==len(deg_2)-1:
                    start = False

        new_coord = coord[node_mask,:]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj)>0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)

    return mod_patch_coord_list, mod_patch_edge_list


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

# ---------------------------
# Resume-safe main (replace original main with this block)
# ---------------------------
import pickle
from pathlib import Path

CHECKPOINT_FILE = "/content/drive/MyDrive/relformer_data/generate_checkpoint.pkl"

def save_checkpoint(stage, data_index, image_id):
    ck = {"stage": stage, "index": int(data_index), "image_id": int(image_id)}
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(ck, f)
    except Exception as e:
        print(f"[WARN] could not save checkpoint: {e}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                ck = pickle.load(f)
                return ck
        except Exception as e:
            print(f"[WARN] could not load checkpoint: {e}")
            return None
    return None

def infer_image_id_from_folder(folder):
    """اگر image_id قبلاً ساخته شده، از بزرگترین شماره فایل‌های raw استخراج کن."""
    try:
        p = Path(folder) / "raw"
        if not p.exists():
            return 1
        files = list(p.glob("sample_*_data.png")) + list(p.glob("sample_*.png"))
        if not files:
            return 1
        # پارس کردن شماره‌ها از نام فایل sample_000001_data.png
        idxs = []
        for f in files:
            name = f.stem  # sample_000001_data
            parts = name.split('_')
            for part in parts:
                if part.isdigit():
                    idxs.append(int(part))
                    break
        if not idxs:
            return 1
        return max(idxs) + 1
    except Exception as e:
        print(f"[WARN] infer_image_id error: {e}")
        return 1

import json

if __name__ == "__main__":
    root_dir = "./data/20cities/"
    checkpoint_file = "./data/checkpoint.json"

    # اگر چک‌پوینت وجود داره، مقدار قبلی رو بخون
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        start_train = checkpoint.get("train_index", 0)
        start_test = checkpoint.get("test_index", 0)
        phase = checkpoint.get("phase", "train")
        print(f"🔄 Resuming from {phase} index {start_train if phase=='train' else start_test}")
    else:
        checkpoint = {}
        start_train = 0
        start_test = 0
        phase = "train"

    image_id = 1

    # ----------------- TRAIN DATA -----------------
    train_path = './data/20cities/train_data/'
    if not os.path.isdir(train_path):
        os.makedirs(train_path+'/seg', exist_ok=True)
        os.makedirs(train_path+'/vtp', exist_ok=True)
        os.makedirs(train_path+'/raw', exist_ok=True)
    print('Preparing Train Data')

    raw_files = [f"{root_dir}/region_{ind}_sat" for ind in indrange_train]
    seg_files = [f"{root_dir}/region_{ind}_gt.png" for ind in indrange_train]
    vtk_files = [f"{root_dir}/region_{ind}_refine_gt_graph.p" for ind in indrange_train]

    # اگر قبلاً train تموم شده بود، از test شروع کن
  if phase == "train":
    for ind in range(start_train, len(raw_files)):
        print(f"Processing train index {ind}")
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

        # ✅ ذخیره چک‌پوینت بعد از هر مرحله
        checkpoint["phase"] = "train"
        checkpoint["train_index"] = ind + 1
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f)

    # بعد از اتمام train، برو سراغ test
    checkpoint["phase"] = "test"
    checkpoint["train_index"] = len(raw_files)
    checkpoint["test_index"] = 0
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)
    phase = "test"


    # ----------------- TEST DATA -----------------
    image_id = 1
    test_path = './data/20cities/test_data/'
    if not os.path.isdir(test_path):
        os.makedirs(test_path+'/seg', exist_ok=True)
        os.makedirs(test_path+'/vtp', exist_ok=True)
        os.makedirs(test_path+'/raw', exist_ok=True)
    print('Preparing Test Data')

    raw_files = [f"{root_dir}/region_{ind}_sat" for ind in indrange_test]
    seg_files = [f"{root_dir}/region_{ind}_gt.png" for ind in indrange_test]
    vtk_files = [f"{root_dir}/region_{ind}_refine_gt_graph.p" for ind in indrange_test]

    start_idx = checkpoint.get("test_index", 0) if phase == "test" else 0
    if phase == "test":
    for ind in range(start_test, len(raw_files)):
        print(f"Processing test index {ind}")
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

        # ✅ ذخیره چک‌پوینت بعد از هر مرحله
        checkpoint["phase"] = "test"
        checkpoint["test_index"] = ind + 1
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f)

    print("✅ All done! Cleaning checkpoint...")
    os.remove(checkpoint_file)
(f"[WARN] could not remove checkpoint: {e}")
