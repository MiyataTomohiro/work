import matplotlib.pyplot as plt
import numpy as np
import os

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

img_rows = 128
img_cols = 128
channels = 3

# 入力画像の次元
img_shape = (img_rows, img_cols, channels)

# 生成器の入力として使われるノイズベクトルの次元
z_dim = 100


def build_generator(img_shape, z_dim):  # 生成器

    model = Sequential()

    # 全結合層
    model.add(Dense(128, input_dim=z_dim))

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # tanh関数を使った出力層
    model.add(Dense(128 * 128 * 3, activation="tanh"))

    # 生成器の出力が画像サイズになるようにreshapeする
    model.add(Reshape(img_shape))

    return model


def build_discriminator(img_shape):  # 識別器
    model = Sequential()

    # 入力画像を一列に並べる
    model.add(Flatten(input_shape=img_shape))

    # 全結合層
    model.add(Dense(128))

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # sigmoid関数を通して出力する
    model.add(Dense(1, activation="sigmoid"))

    return model


def build_gan(generator, discriminator):

    model = Sequential()

    # 生成器と識別器の統合
    model.add(generator)
    model.add(discriminator)

    return model


# 識別器の構築とコンパイル
discriminator = build_discriminator(img_shape)
discriminator.compile(
    loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"]
)

# 生成器の構築
generator = build_generator(img_shape, z_dim)

# 生成器の構築中は識別器のパラメータを固定
discriminator.trainable = False

# 生成器の訓練のため、識別器は固定し、GANモデルの構築とコンパイルを行う
gan = build_gan(generator, discriminator)
gan.compile(loss="binary_crossentropy", optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval, path, output):

    # データセットのロード
    files = np.load(path, allow_pickle=True)
    X_train = files["x_train"]

    # [0, 255]の範囲のグレースケール画素値を[-1, 1]にスケーリング
    X_train = X_train / 127.5 - 1.0

    # 本物の画像のラベルは全て1とする
    real = np.ones((batch_size, 1))

    # 偽の画像のラベルは全て0とする
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  識別器の訓練
        # -------------------------

        # 本物の画像をランダムに取り出したバッチを作る
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # 偽の画像のバッチを生成する
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # 識別器の訓練
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  生成器の訓練
        # ---------------------

        # 偽の画像のバッチを生成する
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # 生成器の訓練
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # 訓練終了後に図示するために、損失と精度をセーブしておく
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # 訓練の進捗を出力する
            print(
                "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                % (iteration + 1, d_loss, 100.0 * accuracy, g_loss)
            )

            # 生成された画像のサンプルを出力する
            sample_images(generator, output, iteration)


def sample_images(generator, path, epoch, image_grid_rows=2, image_grid_columns=2):

    # ランダムノイズのサンプリング
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # ランダムノイズを使って画像を生成する
    gen_imgs = generator.predict(z)

    # 画像の画素値を[0, 1]の範囲にスケールする
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 画像をグリッドに並べる
    fig, axs = plt.subplots(
        image_grid_rows,
        image_grid_columns,
        figsize=(128, 128),
        sharey=True,
        sharex=True,
    )

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 並べた画像を入力する
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
            axs[i, j].axis("off")
            cnt += 1

    fig.savefig(os.path.join(path, f"{epoch}.png"))
    plt.close()


# ハイパラメータの設定
iterations = 20000
batch_size = 128
sample_interval = 1000

# データセット作成
X_train = []
name = input("画像フォルダー名を入力>>")
input_path = os.path.join("/home/miyata/test/img", name)
files = os.listdir(input_path)
output_path = os.path.join("/home/miyata/test/img/res", name)

for target_file in files:
    image = os.path.join(input_path, target_file)
    temp_img = load_img(image, target_size=(128, 128, 3))
    temp_img_arry = img_to_array(temp_img)
    X_train.append(temp_img_arry)

npz_path = os.path.join("/home/miyata/test/npz", f"{name}.npz")
np.savez(npz_path, x_train=X_train)

# 設定した反復回数だけGANの訓練を行う
train(iterations, batch_size, sample_interval, npz_path, output_path)
