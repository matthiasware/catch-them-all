from pathlib import Path
import random
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


from csv_to_tfrecord import csv_to_tfrecord

xy_min = 96
xy_max = 96

xy_min_poke = 16
xy_max_poke = 64
n_classes = 1
n_samples = 100
n_inserts_min = 1
n_inserts_max = 2


p_src = Path("img/poke")
p_dest = Path("img/train_100")
p_dest.mkdir(parents=True, exist_ok=True)
p_annotations = p_dest / "annotations.csv"
p_classes = p_dest / "classes.names"
p_tfrecord = p_dest / "{}.tfrecord".format(p_dest.name)

p_src = str(p_src)
p_dest = str(p_dest)
p_annotations = str(p_annotations)
p_classes = str(p_classes)
p_tfrecord = str(p_tfrecord)

def rand_img(h, w, pil=True):
    dX, dY = w, h
    xArray = np.linspace(0.0, 1.0, dX).reshape((1, dX, 1))
    yArray = np.linspace(0.0, 1.0, dY).reshape((dY, 1, 1))

    def randColor():
        return np.array([random.random(), random.random(), random.random()]).reshape((1, 1, 3))
    def getX(): return xArray
    def getY(): return yArray
    def safeDivide(a, b):
        return np.divide(a, np.maximum(b, 0.001))

    functions = [(0, randColor),
                 (0, getX),
                 (0, getY),
                 (1, np.sin),
                 (1, np.cos),
                 (2, np.add),
                 (2, np.subtract),
                 (2, np.multiply),
                 (2, safeDivide)]
    depthMin = 2
    depthMax = 10

    def buildImg(depth = 0):
        funcs = [f for f in functions if
                    (f[0] > 0 and depth < depthMax) or
                    (f[0] == 0 and depth >= depthMin)]
        nArgs, func = random.choice(funcs)
        args = [buildImg(depth + 1) for n in range(nArgs)]
        return func(*args)

    img = buildImg()

    #import pdb
    #pdb.set_trace()
    # Ensure it has the right dimensions, dX by dY by 3
    img = np.tile(img, (dY // img.shape[0], dX // img.shape[1], 3 // img.shape[2]))

    #pdb.set_trace()
    # Convert to 8-bit, send to PIL and save
    img = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    if pil:
        img = Image.fromarray(img)
    return img

def save_annotations_and_imgs(samples, p_imgs, p_annotations):
    with open(p_annotations, mode='w', newline='') as file:
        csvwriter = csv.writer(file, delimiter=',')
        csvwriter.writerow(["filename", "width", "height",
                            "class", "xmin", "ymin", "xmax", "ymax"])
        for i, (img, bbs) in enumerate(samples):
            p_img = Path(p_imgs) / "img_{}.png".format(i)
            filename = p_img.name
            img.save(str(p_img))
            for bb in bbs:
                csvwriter.writerow([filename,
                                    img.width,
                                    img.height,
                                    bb["class"],
                                    bb["xmin"],
                                    bb["ymin"],
                                    bb["xmax"],
                                    bb["ymax"]])
def save_class_names(p_file, names):
    with open(p_file, mode="w") as file:
        file.write("\n".join(names))
        
def plot_bb(img, bbs, class_map):
    img = np.array(img)
    fig, ax = plt.subplots(1, figsize=(20, 20))
    for i, bb in enumerate(bbs):
        rect = patches.Rectangle(xy=(bb["xmin"], bb["ymin"]),
                                 height=bb["ymax"] - bb["ymin"],
                                 width=bb["xmax"] - bb["xmin"],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.annotate(bb['class'], xy=(bb['xmin'], bb['ymin']))
    ax.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.show()

def create_samples(p_files, xy_min, xy_max, xy_min_poke, xy_max_poke, n_samples, n_inserts_min, n_inserts_max, class_map):
    imgs_poke = [(Image.open(str(p), 'r').convert('RGBA'), p.stem) for p in p_files]

    samples = []
    for idx_sample in range(n_samples):
        xy_size = np.random.randint(xy_min, xy_max + 1)
        n_inserts = np.random.randint(n_inserts_min, n_inserts_max + 1)
        background = rand_img(xy_size, xy_size, pil=True)
        img = Image.new('RGBA', (xy_size, xy_size), (0, 0, 0, 0))
        img.paste(background, (0,0))
        mask = np.zeros((background.height, background.width))
        bbs = []
        for idx_insert in range(n_inserts):
            poke_img, poke_class = imgs_poke[np.random.choice(len(imgs_poke))]
            poke_img = poke_img.copy()
            poke_xy_size = np.random.randint(xy_min_poke, xy_max_poke + 1)
            poke_img.thumbnail((poke_xy_size, poke_xy_size),Image.ANTIALIAS)
            # print(poke_xy_size, poke_img.height, poke_img.width)
            # rot_angle = np.random.randint(0, 360 + 1)
            # poke_img = poke_img.rotate(rot_angle, resample=Image.BICUBIC, expand=True)
            for _ in range(200):
                y_insert = np.random.randint(0, background.height - poke_img.height)
                x_insert = np.random.randint(0, background.width - poke_img.width)
                
                if not np.any(mask[y_insert:(y_insert + poke_img.height), x_insert:(x_insert+poke_img.width)]):
                    #print(np.any(mask[y_insert:poke_img.height, x_insert:poke_img.width]))
                    #print(np.sum(mask[y_insert:poke_img.height, x_insert:poke_img.width]))
                    img.paste(poke_img, (x_insert, y_insert), mask=poke_img.split()[3])
                    mask[y_insert:(y_insert + poke_img.height), x_insert:(x_insert+poke_img.width)] = 1
                    bbs.append({'xmin': x_insert, 'xmax': x_insert + poke_img.width,
                                'ymin': y_insert, 'ymax': y_insert + poke_img.height,
                                'class': class_map[poke_class]})
                    break
            else:
                print("Occupied")
        img.convert("RGB")
    #     plot_bb(img, bbs, class_map)
        samples.append((img, bbs))
    return samples

p_files = list(Path(p_src).glob("*.png"))[::-1]
p_files = p_files if len(p_files) < n_classes else p_files[:n_classes]
class_map = {n: idx for idx, n in enumerate([f.stem for f in p_files])}

samples = create_samples(p_files, xy_min, xy_max, xy_min_poke, xy_max_poke, n_samples, n_inserts_min, n_inserts_max, class_map)
save_annotations_and_imgs(samples, p_dest, p_annotations)
save_class_names(p_classes, list(class_map.keys()))
csv_to_tfrecord(p_dest, p_annotations, p_tfrecord, p_classes)