from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt


def process_img(img_file):
    arr = 255 - np.asarray(Image.open(img_file).convert('L'))
    idx = np.where(arr != 0)
    xl, xr = idx[0][0], idx[0][-1]
    yl, yr = min(idx[1]), max(idx[1])
    xl, xr = xl - (xr - xl) / 10, xr + (xr - xl) / 10
    yl, yr = yl - (yr - yl) / 10, yr + (yr - yl) / 10
    arr = arr[xl:xr, yl:yr]
    arr = np.asarray(Image.fromarray(arr).resize((28, 28), Image.ANTIALIAS)).ravel() / 255.
    return arr


char_map = dict()
classes = range(0, 10) + map(chr, range(65, 91) + range(97, 123))
for k, v in zip(range(1, 63), classes):
    char_map[k] = v

# ____ TRAIN DATA _______
train_data = np.array([])
train_label = np.array([])

# load images
base_dir = 'chars/Sample'
for i in range(1, 63): 
    _dir = base_dir + str(i).zfill(3) + '/'
    base_file = 'img' + str(i).zfill(3) + '-'
    for j in range(1, 51):
        file = base_file + str(j).zfill(3) + '.png'
        # images cropped resized to 28 X 28 
        arr = process_img(_dir + file)
        train_data = np.append(train_data, arr)
        train_label = np.append(train_label, i - 1)


# ____ TEST DATA _______
test_data = np.array([])
test_label = np.array([])

# load images
base_dir = 'chars/Sample'
for i in range(1, 63): 
    _dir = base_dir + str(i).zfill(3) + '/'
    base_file = 'img' + str(i).zfill(3) + '-'
    for j in range(51, 56):
        file = base_file + str(j).zfill(3) + '.png'
        # images cropped resized to 28 X 28 
        arr = process_img(_dir + file)
        test_data = np.append(test_data, arr)
        test_label = np.append(test_label, i - 1)

# save as multi array dict 
train_data = train_data.reshape([50 * 62, -1])
test_data = test_data.reshape([5 * 62, -1])
np.savez('chars.npz', train_data=train_data, train_labels=train_label, 
         test_data=test_data, test_labels=test_label)

