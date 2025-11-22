#!/usr/bin/env python3
# generate_data_resume.py
# Resume-capable version of generate_data.py
# - skip existing outputs
# - auto-detect next image_id
# - robust to missing input files (skip them)
# - safe for long runs in Colab (save to Drive if Drive is mounted and symlinked)

import os
import re
import math
import pickle
import random
import imageio.v2 as imageio
import pyvista
import numpy as np
from pathlib import Path
import sys
import traceback

# -------------------------
# Configuration (change if needed)
# -------------------------
# Default root: script expects data under "<root>/20cities/"
# You can override by setting env var RELFORMER_DATA or by editing root_dir below.
DEFAULT_ROOT = "./data"  # typical project layout: ./data/20cities/
ROOT_ENV = os.environ.get("RELFORMER_DATA", None)
if ROOT_ENV:
    root_dir = os.path.join(ROOT_ENV, "20cities")
else:
    root_dir = os.path.join(DEFAULT_ROOT, "20cities")

patch_size = [128,128,1]
pad = [5,5,0]

# -------------------------
# Utilities
# -------------------------
def angle(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    unit_vector_1 = v1 / n1
    unit_vector_2 = v2 / n2
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
            try:
                idx = node_list.index(nei)
            except ValueError:
                continue
            edge_list.append(np.array((ind,idx)))
    edge_array = np.array(edge_list) if len(edge_list)>0 else np.zeros((0,2), dtype=int)
    return node_array, edge_array

def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):
    """
    path: folder path ending with '/'
    """
    Path(path + "raw/").mkdir(parents=True, exist_ok=True)
    Path(path + "seg/").mkdir(parents=True, exist_ok=True)
    Path(path + "vtp/").mkdir(parents=True, exist_ok=True)

    raw_fn = os.path.join(path, 'raw', f"sample_{str(idx).zfill(6)}_data.png")
    seg_fn = os.path.join(path, 'seg', f"sample_{str(idx).zfill(6)}_seg.png")
    vtp_fn = os.path.join(path, 'vtp', f"sample_{str(idx).zfill(6)}_graph.vtp")

    # If raw already exists, skip (caller should check too)
    if os.path.exists(raw_fn) and os.path.exists(seg_fn) and os.path.exists(vtp_fn):
        print(f"  -> output already exists for id {idx}, skipping write.")
        return False

    # write images
    imageio.imwrite(raw_fn, patch)
    imageio.imwrite(seg_fn, patch_seg)

    # write vtp
    if patch_edge is None or patch_edge.shape[0] == 0 or patch_coord is None or patch_coord.shape[0] == 0:
        # still write an empty vtp with points if possible
        mesh = pyvista.PolyData(patch_coord if patch_coord is not None else np.zeros((0,3)))
        mesh.save(vtp_fn)
    else:
        # convert edges to VTP lines format
        patch_edge2 = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
        mesh = pyvista.PolyData(patch_coord)
        mesh.lines = patch_edge2.flatten()
        mesh.save(vtp_fn)
    return True

def find_next_image_id(save_path):
    """Scan save_path/raw for existing sample_######_data.png and return next id (1-based)"""
    raw_dir = os.path.join(save_path, "raw")
    if not os.path.isdir(raw_dir):
        return 1
    pattern = re.compile(r"sample_(\d+)_data\.png")
    max_id = 0
    for fn in os.listdir(raw_dir):
        m = pattern.match(fn)
        if m:
            val = int(m.group(1))
            if val > max_id:
                max_id = val
    return max_id + 1

# prune_patch & patch_extract reimplemented with safer checks

def prune_patch(patch_coord_list, patch_edge_list):
    mod_patch_coord_list = []
    mod_patch_edge_list = []
    for coord, edge in zip(patch_coord_list, patch_edge_list):
        if coord is None or coord.shape[0] == 0 or edge is None or edge.shape[0] == 0:
            continue
        # adjacency by squared euclidean distance
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:,0], edge[:,1]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)
        dist_adj[edge[:,1], edge[:,0]] = dist_adj[edge[:,0], edge[:,1]]

        start = True
        node_mask = np.ones(coord.shape[0], dtype=bool)
        while start:
            degree = (dist_adj>0).sum(1)
            deg_2 = list(np.where(degree==2)[0])
            if len(deg_2)==0:
                start = False
                break
            removed_any = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx,:]>0)[0]
                if deg_2_neighbor.shape[0] != 2:
                    continue
                p1 = coord[idx,:]
                p2 = coord[deg_2_neighbor[0],:]
                p3 = coord[deg_2_neighbor[1],:]
                l1 = p2-p1
                l2 = p3-p1
                node_angle = angle(l1,l2)*180 / math.pi
                if node_angle>160:
                    node_mask[idx]=False
                    dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]] = np.sum((p2-p3)**2)
                    dist_adj[deg_2_neighbor[1], deg_2_neighbor[0]] = dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]]
                    dist_adj[idx, deg_2_neighbor[0]] = 0.0
                    dist_adj[deg_2_neighbor[0], idx] = 0.0
                    dist_adj[idx, deg_2_neighbor[1]] = 0.0
                    dist_adj[deg_2_neighbor[1], idx] = 0.0
                    removed_any = True
                    break
                elif n == len(deg_2) - 1:
                    # finished checking all deg2 nodes and removed none
                    start = False
            if not removed_any and start:
                start = False

        new_coord = coord[node_mask,:]
        if new_coord.shape[0] < 1:
            continue
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj)>0)).T
        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)
    return mod_patch_coord_list, mod_patch_edge_list

def patch_extract(save_path,image, seg,  mesh, image_id_start=1):
    """
    Extract patches from one image+mesh and save to save_path.
    Returns next image_id after saved items.
    """
    image_id = image_id_start
    p_h, p_w, _ = patch_size
    pad_h, pad_w, _ = pad
    p_h = p_h - 2*pad_h
    p_w = p_w - 2*pad_w

    h, w, d = image.shape
    x_ = np.int32(np.linspace(5, max(5, h-5-p_h), 32))
    y_ = np.int32(np.linspace(5, max(5, w-5-p_w), 32))
    ind = np.meshgrid(x_, y_, indexing='ij')

    for i, start in enumerate(list(np.array(ind).reshape(2,-1).T)):
        start = np.array((start[0], start[1], 0))
        end = start + np.array(patch_size)-1 - 2*np.array(pad)
        # bounds for clipping
        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]
        try:
            clipped_mesh = mesh.clip_box(bounds, invert=False)
        except Exception as e:
            # mesh clipping might fail for empty mesh
            continue
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        # cells: lines encoded after point/cell headers: guard for empty
        try:
            patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes==1)*2:].reshape(-1,3)
        except:
            patch_edge = np.zeros((0,3), dtype=int)

        # filter coords inside patch bounds
        if patch_coordinates.shape[0] == 0:
            continue
        inside_mask = (np.prod(patch_coordinates >= start, 1) * np.prod(patch_coordinates <= end, 1)) > 0.0
        patch_coordinates = patch_coordinates[inside_mask, :]
        if patch_coordinates.shape[0] < 2:
            continue

        # remap edges: keep only edges whose endpoints are inside the coords
        if patch_edge.shape[0] == 0:
            continue
        candidate_edges = [tuple(l[1:]) for l in patch_edge if l.shape[0] >= 3]
        # map global indices to local
        global_idx = np.where(inside_mask)[0]
        flat = []
        for e in candidate_edges:
            if e[0] in global_idx and e[1] in global_idx:
                # remap indices
                local0 = int(np.where(global_idx == e[0])[0])
                local1 = int(np.where(global_idx == e[1])[0])
                flat.append((local0, local1))
        if len(flat) == 0:
            continue
        patch_edge_arr = np.array(flat, dtype=int)
        # normalize coordinates to patch local coordinates scaled by patch_size
        patch_coordinates_local = (patch_coordinates - start + np.array(pad)) / np.array(patch_size)

        # prune
        mod_patch_coord_list, mod_patch_edge_list = prune_patch([patch_coordinates_local], [patch_edge_arr])
        if len(mod_patch_coord_list) == 0:
            continue

        # patch image and seg
        try:
            patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :], ((pad_h,pad_h),(pad_w,pad_w),(0,0)))
            patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w], ((pad_h,pad_h),(pad_w,pad_w)))
        except Exception as e:
            continue

        for patch_img, patch_mask, patch_coord, patch_edge in zip([patch], [patch_seg], mod_patch_coord_list, mod_patch_edge_list):
            if patch_mask.sum() > 10:
                written = save_input(save_path, image_id, patch_img, patch_mask, patch_coord, patch_edge)
                if written:
                    print(f"    saved id {image_id}")
                image_id += 1
    return image_id

# -------------------------
# Main flow (resume-capable)
# -------------------------
def list_regions(root_dir):
    """Return sorted list of region indices based on files like region_{i}_sat.png or .jpg"""
    files = []
    if not os.path.isdir(root_dir):
        return []
    for fn in os.listdir(root_dir):
        # expecting e.g. region_12_sat.png or region_12_sat.jpg or region_12_gt.png or region_12_refine_gt_graph.p
        if "_sat" in fn:
            files.append(fn)
    # derive indices
    inds = []
    for fn in files:
        m = re.search(r"region_(\d+)_sat", fn)
        if m:
            inds.append(int(m.group(1)))
    return sorted(list(set(inds)))

def build_index_lists():
    # old scheme from repo: derive indrange_train and indrange_test from 0..179
    indrange_train = []
    indrange_test = []
    for x in range(180):
        if x % 10 < 8:
            indrange_train.append(x)
        if x % 10 == 9:
            indrange_test.append(x)
        if x % 20 == 18:
            indrange_train.append(x)
        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_test

def file_exists_any_ext(basepath_noext):
    # check for .png or .jpg presence
    for ext in [".png", ".jpg", ".jpeg"]:
        if os.path.exists(basepath_noext + ext):
            return basepath_noext + ext
    return None

def process_set(ind_list, save_path, set_name):
    print(f"Processing {set_name}: saving to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "seg"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "vtp"), exist_ok=True)

    next_id = find_next_image_id(save_path)
    print(f"  next image_id = {next_id}")

    for idx_region in ind_list:
        base = os.path.join(root_dir, f"region_{idx_region}")
        sat_path = file_exists_any_ext(base + "_sat")
        if sat_path is None:
            print(f"  [skip] sat missing for region {idx_region}: looked for {base}_sat.png/.jpg")
            continue
        seg_path = file_exists_any_ext(base + "_gt")
        if seg_path is None:
            print(f"  [skip] seg missing for region {idx_region}: looked for {base}_gt.png/.jpg")
            continue
        vtk_path = file_exists_any_ext(base + "_refine_gt_graph.p")
        # for p files we accept exact .p only
        if vtk_path is None:
            pth = base + "_refine_gt_graph.p"
            if not os.path.exists(pth):
                print(f"  [skip] vtk/pickle missing for region {idx_region}: {pth}")
                continue
            else:
                vtk_path = pth

        # read inputs, robustly
        try:
            sat_img = imageio.imread(sat_path)
        except Exception as e:
            print(f"  [skip] failed to read sat for {idx_region}: {e}")
            continue
        try:
            with open(vtk_path, "rb") as f:
                graph = pickle.load(f)
        except Exception as e:
            print(f"  [skip] failed to load graph for {idx_region}: {e}")
            continue
        try:
            gt_seg = imageio.imread(seg_path)
        except Exception as e:
            print(f"  [skip] failed to read seg for {idx_region}: {e}")
            continue

        # convert graph->nodes,edges
        node_array, edge_array = convert_graph(graph)
        if node_array is None or node_array.shape[0] == 0:
            print(f"  [skip] empty node array for region {idx_region}")
            continue

        patch_coord = np.concatenate((node_array, np.int32(np.zeros((node_array.shape[0],1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        if edge_array is None or edge_array.shape[0] == 0:
            patch_edge = np.zeros((0,2), dtype=int)
        else:
            patch_edge = np.concatenate((np.int32(2*np.ones((edge_array.shape[0],1))), edge_array), 1)
            mesh.lines = patch_edge.flatten()

        # make sure gt_seg is 2D grayscale
        if gt_seg.ndim == 3:
            # assume RGB -> take one channel or convert to grayscale if necessary
            gt_seg = gt_seg[...,0]
        # normalize seg (0/1)
        try:
            if np.max(gt_seg) > 1:
                gt_seg = (gt_seg / np.max(gt_seg)).astype(np.uint8)
        except:
            pass

        # call patch extractor (it will write files)
        try:
            next_id = patch_extract(save_path, sat_img, gt_seg, mesh, image_id_start=next_id)
        except Exception as e:
            print(f"  [error] patch_extract failed for region {idx_region}: {e}")
            traceback.print_exc()
            continue

    print(f"Finished {set_name}. next image_id would be {next_id}")

def main():
    print("Resume-capable generate_data starting")
    print("Root dir (where '20cities' expected):", root_dir)
    indrange_train, indrange_test = build_index_lists()

    # process train
    train_save = os.path.join(root_dir, "train_data")
    # if train_save exists, don't error — we'll resume/skip
    process_set(indrange_train, train_save, "Train Data")

    # process test
    test_save = os.path.join(root_dir, "test_data")
    process_set(indrange_test, test_save, "Test Data")

    print("All done.")

if __name__ == "__main__":
    main()
