import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

# 画像のパラメータ(サイズ、チャンネル数)
img_rows = 128
img_cols = 128
channels = 3

# 入力画像の次元
img_shape = (img_rows, img_cols, channels)


# 画像を表示する関数
def show_imgs(imgs, row, col):
    # 画像からなるグリッドを生成する
    fig, axs = plt.subplots(
        row,
        col,
        figsize=(32, 32),
        sharey=True,
        sharex=True,
    )

    cnt = 0
    for i in range(row):
        for j in range(col):
            # 画像グリッドを表示する
            axs[i, j].imshow(imgs[cnt])
            axs[i, j].axis("off")
            cnt += 1

    fig.savefig(os.path.join(dataset_path, f"{name}.png"))
    plt.close()


def train(iterations, batch_size, sample_interval):

    # データセットのロード
    images = []
    img_list = os.listdir(input_path)
    for img in img_list:
        image = img_to_array(
            load_img(os.path.join(input_path, img), target_size=img_shape)
        )
        # -1から1の範囲に正規化
        image = (image.astype(np.float32) - 127.5) / 127.5
        images.append(image)

    # 画像の表示
    show_imgs(images, 5, 8)


# ハイパラメータの設定
iterations = 30000
batch_size = 40
sample_interval = 1000

# 入出力パスの設定
name = input("画像フォルダー名を入力>>")
input_path = os.path.join("/home/miyata/test/img", name)
dataset_path = os.path.join("/home/miyata/test/img/res/dataset", name)

# 決められた回数だけDCGANの訓練を反復する
train(iterations, batch_size, sample_interval)
