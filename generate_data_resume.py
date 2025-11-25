import os
import imageio
import pickle
import numpy as np
import pyvista
import cv2
from tqdm import tqdm


#############################################################
# ابزارهای کمکی
#############################################################

def safe_mkdir(path):
    """ساخت پوشه فقط اگر وجود نداشته باشد"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def find_next_image_id(root_dir):
    """یافتن شماره بعدی نمونه برای ادامه دادن تولید دیتا"""
    raw_dir = os.path.join(root_dir, "raw")
    safe_mkdir(raw_dir)

    existing = [
        fn for fn in os.listdir(raw_dir)
        if fn.startswith("sample_") and fn.endswith("_data.png")
    ]

    if len(existing) == 0:
        return 0

    ids = []
    for f in existing:
        # sample_000123_data.png
        try:
            num = int(f.split("_")[1])
            ids.append(num)
        except:
            pass

    if len(ids) == 0:
        return 0
    return max(ids) + 1


def save_input(path, idx, patch, patch_seg, coord, edge):
    """ذخیره یک پچ"""
    raw_path = os.path.join(path, "raw")
    seg_path = os.path.join(path, "seg")
    vtp_path = os.path.join(path, "vtp")

    safe_mkdir(raw_path)
    safe_mkdir(seg_path)
    safe_mkdir(vtp_path)

    imageio.imwrite(f"{raw_path}/sample_{idx:06d}_data.png", patch)
    imageio.imwrite(f"{seg_path}/sample_{idx:06d}_seg.png", patch_seg)

    mesh = pyvista.PolyData(coord)
    mesh.lines = np.concatenate(
        (np.full((edge.shape[0], 1), 2, dtype=np.int32), edge), axis=1
    ).flatten()

    mesh.save(f"{vtp_path}/sample_{idx:06d}.vtp")


#############################################################
# ابزار استخراج پچ
#############################################################

PATCH_SIZE = 512

def patch_extract(out_dir, sat, seg, mesh, start_id):

    H, W = sat.shape[:2]
    step = PATCH_SIZE

    next_id = start_id

    for y in range(0, H - PATCH_SIZE, step):
        for x in range(0, W - PATCH_SIZE, step):

            sat_patch = sat[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            seg_patch = seg[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # استخراج مختصات نودها داخل پچ
            pts = mesh.points
            mask = (
                (pts[:, 0] >= x) & (pts[:, 0] < x + PATCH_SIZE) &
                (pts[:, 1] >= y) & (pts[:, 1] < y + PATCH_SIZE)
            )
            pts_sub = pts[mask]

            # اگر هیچ نودی داخل پچ نیست → رد شود
            if len(pts_sub) == 0:
                continue

            # تبدیل مختصات نسبی
            pts_sub_rel = pts_sub.copy()
            pts_sub_rel[:, 0] -= x
            pts_sub_rel[:, 1] -= y

            # تصمیم لبه ها
            lines = mesh.lines.reshape(-1, 3)
            edges = []
            for L, a, b in lines:
                if mask[a] and mask[b]:
                    edges.append([np.where(mask)[0].tolist().index(a),
                                  np.where(mask)[0].tolist().index(b)])

            if len(edges) == 0:
                continue

            edges = np.array(edges, dtype=np.int32)

            save_input(out_dir, next_id, sat_patch, seg_patch, pts_sub_rel, edges)
            next_id += 1

    return next_id


#############################################################
# تبدیل گراف های pickle شده
#############################################################

def convert_graph(graph):
    """ورودی: گراف pickle شده  
       خروجی: node_array و edge_array (برای ساخت mesh)"""

    nodes = []
    edges = []

    node_id_map = {}

    for i, (nid, item) in enumerate(graph['nodes'].items()):
        nodes.append([item['x'], item['y'], 0])
        node_id_map[nid] = i

    for (s, e) in graph['edges']:
        if s in node_id_map and e in node_id_map:
            edges.append([node_id_map[s], node_id_map[e]])

    return np.array(nodes, np.int32), np.array(edges, np.int32)


#############################################################
# شروع اصلی برنامه
#############################################################

def process_set(ind_range, save_path, root_dir, label):

    print(f"\n===== Processing {label} =====")

    safe_mkdir(save_path)
    safe_mkdir(os.path.join(save_path, "raw"))
    safe_mkdir(os.path.join(save_path, "seg"))
    safe_mkdir(os.path.join(save_path, "vtp"))

    next_id = find_next_image_id(save_path)
    print(f"Continue from ID = {next_id}")

    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in ind_range:
        raw_files.append(f"{root_dir}/region_{ind}_sat")
        seg_files.append(f"{root_dir}/region_{ind}_gt.png")
        vtk_files.append(f"{root_dir}/region_{ind}_refine_gt_graph.p")

    for i in range(len(raw_files)):
        print(f"Region {ind_range[i]} ({i+1}/{len(raw_files)})")

        try:
            sat = imageio.imread(raw_files[i] + ".png")
        except:
            sat = imageio.imread(raw_files[i] + ".jpg")

        seg = imageio.imread(seg_files[i])

        with open(vtk_files[i], "rb") as f:
            g = pickle.load(f)

        nodes, edges = convert_graph(g)

        mesh = pyvista.PolyData(np.concatenate((nodes, np.zeros((nodes.shape[0],1))), axis=1))
        mesh.lines = np.concatenate((np.full((edges.shape[0],1),2), edges), axis=1).flatten()

        next_id = patch_extract(save_path, sat, seg, mesh, next_id)

    print(f"Finished {label}, next ID = {next_id}")


#############################################################
# main
#############################################################

def main():

    root_dir = "./data/20cities"

    # رنج منطقه‌ها
    indrange_train = list(range(0, 90))
    indrange_test = list(range(90, 100))

    train_dir = "./data/20cities/train_data"
    test_dir  = "./data/20cities/test_data"

    process_set(indrange_train, train_dir, root_dir, "Train Data")
    process_set(indrange_test, test_dir, root_dir, "Test Data")


if __name__ == "__main__":
    print("\nResume-capable generate_data starting...\n")
    main()
