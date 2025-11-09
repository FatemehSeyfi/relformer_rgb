import os
import imageio.v2 as imageio
from tqdm import tqdm
import numpy as np
from loguru import logger

# -------------------------------
# مسیر داده‌ها
# -------------------------------
DATA_ROOT = "data/20cities"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# -------------------------------
# تنظیمات
# -------------------------------
RAW_FILES = [f"region_{i}" for i in range(200)]  # تعداد مناطق فرضی (تغییر بده بر اساس پروژه‌ات)
SAVE_EVERY = 5  # هر چند نمونه، وضعیت ذخیره بشه
STATUS_FILE = os.path.join(DATA_ROOT, "progress.txt")

# -------------------------------
# خواندن آخرین وضعیت
# -------------------------------
def load_progress():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_progress(i):
    with open(STATUS_FILE, "w") as f:
        f.write(str(i))

# -------------------------------
# تابع ساخت داده (مثال)
# -------------------------------
def process_region(region_id):
    # ورودی‌ها
    base_path = os.path.join(DATA_ROOT, f"region_{region_id}_sat.png")
    out_path = os.path.join(TRAIN_DIR, f"region_{region_id}_processed.npy")

    # اگر خروجی قبلاً هست، رد شو
    if os.path.exists(out_path):
        logger.info(f"✅ Region {region_id} already processed — skipping.")
        return

    if not os.path.exists(base_path):
        logger.warning(f"⚠️ Input image not found: {base_path}")
        return

    # خواندن تصویر
    img = imageio.imread(base_path)
    np.save(out_path, img.mean(axis=-1))  # فقط مثال از پردازش

# -------------------------------
# اجرای مقاوم در برابر قطع شدن
# -------------------------------
start_index = load_progress()
logger.info(f"▶️ Resuming from region {start_index}")

for i in tqdm(range(start_index, len(RAW_FILES))):
    try:
        process_region(i)
    except Exception as e:
        logger.error(f"❌ Error at region {i}: {e}")
        break
    if i % SAVE_EVERY == 0:
        save_progress(i)

# ذخیره آخرین وضعیت
save_progress(len(RAW_FILES))
logger.info("🏁 All regions processed successfully.")
