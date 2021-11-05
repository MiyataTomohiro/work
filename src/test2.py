import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Flatten,
    Reshape,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

# img_rows = 128
# img_cols = 128
img_rows = 256
img_cols = 256
channels = 3

# 入力画像の次元
img_shape = (img_rows, img_cols, channels)

# 生成器の入力として使われるノイズベクトルのサイズ
z_dim = 100


def build_generator(z_dim):

    model = Sequential()

    # 全結合層によって、32×32×256のテンソルに変換
    model.add(Dense(256 * 32 * 32, input_dim=z_dim))
    model.add(Reshape((32, 32, 256)))

    # 転置畳み込み層により、32×32×256を64×64×128のテンソルに変換
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLUによる活性化
    model.add(LeakyReLU(alpha=0.01))

    # 転置畳み込み層により、64×64×128を64×64×64のテンソルに変換
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding="same"))

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLUによる活性化
    model.add(LeakyReLU(alpha=0.01))

    # 転置畳み込み層により、64×64×64を128×128×3のテンソルに変換
    model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding="same"))

    # 転置畳み込み層により、128×128×3を256×256×3のテンソルに変換
    model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding="same"))

    # tanh関数を適用して出力
    model.add(Activation("tanh"))

    return model


def build_discriminator(img_shape):

    model = Sequential()

    # 256×256×3を128×128×3のテンソルにする畳み込み層
    model.add(
        Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
    )

    # 128×128×3を64×64×32のテンソルにする畳み込み層
    model.add(
        Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
    )

    # Leaky ReLUによる活性化
    model.add(LeakyReLU(alpha=0.01))

    # 64×64×32を32×32×64のテンソルにする畳み込み層
    model.add(
        Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
    )

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLUによる活性化
    model.add(LeakyReLU(alpha=0.01))

    # 32×32×64を16×16×128のテンソルにする畳み込み層
    model.add(
        Conv2D(128, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
    )

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLUによる活性化
    model.add(LeakyReLU(alpha=0.01))

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


# 画像を表示する関数
def show_imgs(imgs, row, col, file_name):
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

    fig.savefig(os.path.join(dataset_path, f"{file_name}.png"))
    plt.close()


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

    # # matplotlibでロードした画像の表示して確認
    # show_imgs(X_train, 5, 8)
    
    # 4Dテンソルに変換(データの個数, 128, 128, 3)
    X_train = np.array(X_train)

    # 本物の画像のラベルは全て1にする
    real = np.ones((batch_size, 1))

    # 偽物の画像のラベルは全て0にする
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        print(iteration)
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
            # matplotlibでロードした画像の表示して確認
            show_imgs(X_train, 5, 8, iteration)
            # # 本物の画像を取り出して、保存する
            # img = Image.fromarray(np.uint8((imgs[0] + 127.5) * 127.5))
            # img.save(os.path.join(real_path, f"real_{iteration}.png"))
            # # 偽物の画像を取り出して、保存する
            # gan_img = Image.fromarray(np.uint8((gen_imgs[0] + 127.5) * 127.5))
            # gan_img.save(os.path.join(fake_path, f"fake_{iteration}.png"))

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
        figsize=(32, 32),
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
# iterations = 30000
# batch_size = 128
iterations = 30000
batch_size = 40
sample_interval = 1000

# 入出力パスの設定
name = input("画像フォルダー名を入力>>")
input_path = os.path.join("/home/miyata/test/img", name)
output_path = os.path.join("/home/miyata/test/img/res", name)
# real_path = os.path.join("/home/miyata/test/img/real", name)
# fake_path = os.path.join("/home/miyata/test/img/fake", name)
dataset_path = os.path.join("/home/miyata/test/img/res/dataset", name)

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
