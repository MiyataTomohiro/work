import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img, img_to_array, load_img

img_rows = 128
img_cols = 128
channels = 3

# 入力画像の次元
img_shape = (img_rows, img_cols, channels)

# 生成器の入力として使われるノイズベクトルのサイズ
z_dim = 100


def build_generator(z_dim):

    model = Sequential()

    model.add(Dense(128 * 32 * 32, input_dim=z_dim))
    model.add(Reshape((32, 32, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(3, kernel_size=3, padding="same"))

    # tanh関数を適用して出力
    model.add(Activation("tanh"))

    return model


def build_discriminator(img_shape):

    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
    )
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # 出力にシグモイド関数を適用
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model


def build_gan(generator, discriminator):

    model = Sequential()

    # 生成器と識別器のモデルを組み合わせる
    model.add(generator)
    model.add(discriminator)

    return model


# 識別器の構築とコンパイル
discriminator = build_discriminator(img_shape)
discriminator.compile(
    loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"]
)

# 生成器の構築
generator = build_generator(z_dim)

# 生成器の訓練時には、識別器のパラメータは定数にする
discriminator.trainable = False

# 識別器は固定したまま、生成器を訓練するGANモデルの生成とコンパイル
gan = build_gan(generator, discriminator)
gan.compile(loss="binary_crossentropy", optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):

    # データセットのロード
    X_train = []
    img_list = os.listdir(input_path)
    for img in img_list:
        image = img_to_array(
            load_img(os.path.join(input_path, img), target_size=img_shape)
        )
        # -1から1の範囲に正規化
        image = (image.astype(np.float32) - 127.5) / 127.5
        X_train.append(image)

    # 4Dテンソルに変換(データの個数, 128, 128, 3)
    X_train = np.array(X_train)

    # 本物の画像のラベルは全て1にする
    real = np.ones((batch_size, 1))

    # 偽物の画像のラベルは全て0にする
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  識別器の学習
        # -------------------------

        # 本物の画像集合からランダムにバッチを生成する
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # 偽物の画像からなるバッチを生成する
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        if (iteration + 1) % sample_interval == 0:
            # 本物の画像を取り出して、保存する
            img = Image.fromarray(np.uint8((imgs[0] + 127.5) * 127.5))
            img.save(os.path.join(real_path, f"real_{iteration}.png"))
            # 偽物の画像を取り出して、保存する
            gan_img = Image.fromarray(np.uint8((gen_imgs[0] + 127.5) * 127.5))
            gan_img.save(os.path.join(fake_path, f"fake_{iteration}.png"))

        # 識別器の学習
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  生成器の学習
        # ---------------------

        # 偽物の画像からなるバッチを生成する
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # 生成器の学習
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # あとで可視化するために損失と精度を保存しておく
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # 学習結果の出力
            print(
                "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                % (iteration + 1, d_loss, 100.0 * accuracy, g_loss)
            )

            # 生成したサンプル画像を出力する
            sample_images(generator, iteration)


def sample_images(generator, epoch, image_grid_rows=10, image_grid_columns=10):

    # ノイズベクトルを生成する
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # ノイズベクトルから画像を生成する
    gen_imgs = generator.predict(z)

    # 出力の画素値を[0, 1]の範囲にスケーリングする
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 画像からなるグリッドを生成する
    fig, axs = plt.subplots(
        image_grid_rows,
        image_grid_columns,
        figsize=(img_rows, img_cols),
        sharey=True,
        sharex=True,
    )

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 画像グリッドを表示する
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis("off")
            cnt += 1

    fig.savefig(os.path.join(output_path, f"{epoch}.png"))
    plt.close()


# ハイパラメータの設定
iterations = 100000
batch_size = 32
sample_interval = 1000

# 入出力パスの設定
name = input("画像フォルダー名を入力>>")
input_path = os.path.join("/home/miyata/test/img", name)
output_path = os.path.join("/home/miyata/test/img/res", name)
real_path = os.path.join("/home/miyata/test/img/real", name)
fake_path = os.path.join("/home/miyata/test/img/fake", name)

# 決められた回数だけDCGANの訓練を反復する
train(iterations, batch_size, sample_interval)

losses = np.array(losses)

# 生成器と識別器の学習損失をプロット
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_path, "Training Loss.png"))


accuracies = np.array(accuracies)

# 識別器の精度をプロット
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")

plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig(os.path.join(output_path, "Discriminator Accuracy.png"))
