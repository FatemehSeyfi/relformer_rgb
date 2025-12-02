import pdb
import math
import imageio.v2 as imageio  # اصلاح برای DeprecationWarning
import pyvista
import numpy as np
import pickle
import os

patch_size = [128,128,1]
pad = [5,5,0]

def find_next_image_id(path):
    raw_dir = os.path.join(path, "raw")
    if not os.path.exists(raw_dir):
        return 1
    files = [f for f in os.listdir(raw_dir) if f.endswith("_data.png")]
    if len(files) == 0:
        return 1
    nums = [int(f.split("_")[1]) for f in files]
    return max(nums) + 1

def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min=-1, a_max=1))

def convert_graph(graph):
    node_list = []
    edge_list = []
    for n, v in graph.items():
        node_list.append(n)
    node_array = np.array(node_list)

    for ind, (n, v) in enumerate(graph.items()):
        for nei in v:
            idx = node_list.index(nei)
            edge_list.append(np.array((ind, idx)))
    edge_array = np.array(edge_list)
    return node_array, edge_array

def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):
    raw_path = os.path.join(path, "raw", f"sample_{str(idx).zfill(6)}_data.png")
    seg_path = os.path.join(path, "seg", f"sample_{str(idx).zfill(6)}_seg.png")
    vtp_path = os.path.join(path, "vtp", f"sample_{str(idx).zfill(6)}_graph.vtp")

    if os.path.exists(raw_path) and os.path.exists(seg_path) and os.path.exists(vtp_path):
        print(f"[SKIP] sample {idx} already exists.")
        return

    imageio.imwrite(raw_path, patch)
    imageio.imwrite(seg_path, patch_seg)

    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    mesh.lines = patch_edge.flatten()
    mesh.save(vtp_path)
    print(f"[DONE] sample {idx} created.")
    
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


# patch_extract و prune_patch همان نسخه قبل

# --- Resume-Capable Main ---
if __name__ == "__main__":
    root_dir = "./data/20cities/"

    train_path = os.path.join(root_dir, "train_data")
    test_path = os.path.join(root_dir, "test_data")

    # ایجاد پوشه‌ها در صورت عدم وجود
    for path in [train_path, test_path]:
        for sub in ["raw","seg","vtp"]:
            os.makedirs(os.path.join(path, sub), exist_ok=True)

    # تعیین ادامه ID
    image_id = find_next_image_id(train_path)
    print(f"Resume Train Data from ID = {image_id}")

    # آماده سازی لیست فایل‌ها
    indrange_train = [x for x in range(180) if x%10<8 or x%20==18]
    indrange_test  = [x for x in range(180) if x%10==9 or x%20==8]

    # --- Train ---
    for ind in indrange_train:
        try:
            sat_img = imageio.imread(os.path.join(root_dir, f"region_{ind}_sat.png"))
        except:
            sat_img = imageio.imread(os.path.join(root_dir, f"region_{ind}_sat.jpg"))
        with open(os.path.join(root_dir, f"region_{ind}_refine_gt_graph.p"), 'rb') as f:
            graph = pickle.load(f)
        node_array, edge_array = convert_graph(graph)
        gt_seg = imageio.imread(os.path.join(root_dir, f"region_{ind}_gt.png"))

        patch_coord = np.concatenate((node_array, np.zeros((node_array.shape[0],1),dtype=np.int32)),1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate((np.int32(2*np.ones((edge_array.shape[0],1))), edge_array),1)
        mesh.lines = patch_edge.flatten()

        patch_extract(train_path, sat_img, gt_seg, mesh)

    # --- Test ---
    image_id = find_next_image_id(test_path)
    print(f"Resume Test Data from ID = {image_id}")

    for ind in indrange_test:
        try:
            sat_img = imageio.imread(os.path.join(root_dir, f"region_{ind}_sat.png"))
        except:
            sat_img = imageio.imread(os.path.join(root_dir, f"region_{ind}_sat.jpg"))
        with open(os.path.join(root_dir, f"region_{ind}_refine_gt_graph.p"), 'rb') as f:
            graph = pickle.load(f)
        node_array, edge_array = convert_graph(graph)
        gt_seg = imageio.imread(os.path.join(root_dir, f"region_{ind}_gt.png"))

        patch_coord = np.concatenate((node_array, np.zeros((node_array.shape[0],1),dtype=np.int32)),1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate((np.int32(2*np.ones((edge_array.shape[0],1))), edge_array),1)
        mesh.lines = patch_edge.flatten()

        patch_extract(test_path, sat_img, gt_seg, mesh)
