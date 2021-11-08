# DCGANのクラス(class)を作成
class DCGAN:
    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
    クラス : モノを作るときの設計図
    インスタンス : 設計図から実際に作られたモノ
    self :  クラス構造を取る際の定型の構文(インスタンス自身, その時点の自分, メソッドの仮引数)
    __init__() :    コンストラクタと呼ばれる初期化のための関数(メソッド)
                    インスタンス化を行う時に必ず最初に呼び出される特殊な関数(メソッド)
                    オブジェクト生成(インスタンスを生成)のついでにデータの初期化を行うもの
    """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

    def __init__(self):
        # クラス分類用のクラス名
        self.class_names = os.listdir(root_dir)

        # 入力画像サイズ(shape)と潜在変数の次元(z_dim)
        self.shape = (128, 128, 3)
        self.z_dim = 100

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
        optimizer : オプティマイザー(最適化アルゴリズム)
        Adam  --> lr : 学習率(0以上の浮動小数点数)
                  beta_1 : 一般的に1に近い値(0<beta<1の浮動小数点数)
        """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
        optimizer = Adam(lr=0.0002, beta_1=0.5)

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
        識別器(Discriminator)はRealかfakeを見分ける二値分類を行うためbinary_crossentropyを使用
        
        クロスエントロピー : 一種の距離を表す指標(2つの値がどれだけ離れているかを示す尺度)
        binary_crossentropy : バイナリクロスエントロピー
        [0,1]をとる変数と2クラスラベルにベルヌーイ分布を仮定した場合の対数尤度

        compile(コンパイル) : モデルの学習を始める前に、どのような学習処理を実行するか設定
        loss(損失関数) : 　モデルが最小化しようとする目的関数
        optimizer(最適化アルゴリズム) : Optimizerクラスのインスタンスを与える
        metrics(評価関数のリスト) : 分類問題では精度としてmetrics=['accuracy']を指定
        """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
        self.discriminator = self.build_discriminator()
        self.discrinator.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
        生成器(Generator)学習用の組合せモデル(Combined_model)を作成
        Input() : Kerasテンソルのインスタンス化に使用

        識別器(Discriminator)のパラメータは固定
        --> .trainable = False のように設定するとパラメータが学習中に更新されない
        """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
        self.generator = self.build_generator()
        z = Input(shape=(100,))  # 入力値(z) : 潜在変数(Noise)
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)  # 出力値(Valid): Real(=1), Fake(=0)を取る

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
        Model() : テンソル(多次元配列)の入出力を与え, モデルをインスタンス化
        compile(コンパイル) : モデルの学習を始める前に、どのような学習処理を実行するか設定
        loss(損失関数) : 　モデルが最小化しようとする目的関数
        --> combined_model(Generatorの学習用モデル)も二値分類を行うためbinary_crossentropy
        optimizer(最適化アルゴリズム) : Optimizerクラスのインスタンスを与える
        """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def build_combined(self):
        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """
        build_combined() : 生成器(Generator)学習用の組合せモデル(Combined_model)を作成する関数

        識別器(Discriminator)のパラメータは固定
        --> .trainable = False のように設定するとパラメータが学習中に更新されない

        Sequential() : モデル層を積み重ねる形式の記述方法
        """ """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])

    def train(
        self,
        iterations=200000,
        batch_size=32,
        save_interval=1000,
        model_interval=5000,
    ):
        """
        train() : DCGANの生成器と識別器が学習する関数
        epoches(エポック数) :
        batch_size(バッチサイズ) : データセットをいくつかのサブセットに分けた時にに含まれるデータの数
        --> 2のn乗の値が使われることが多く, [32, 64, 128, 256, 512, 1024, 2048]などがよく使われる数値
        save_interval :

        X_train : 学習に使用するデータセットの画像データ
        labels : 学習に使用するデータセットのラベル
        half_batch : batch_sizeの半分

        """
        X_train, labels = self.load_imgs()

        half_batch = int(batch_size / 2)

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        for epoch in range(epochs):

            """""" """""" """""" """""" """""" """
            識別器の訓練(Training Generator)
            
            """ """""" """""" """""" """""" """"""
            idx = np.random.randint(0, X_train.shape[0], half_batch)

            imgs = X_train[idx]

            noise = np.random.uniform(-1, 1, (half_batch, self.z_dim))

            gen_imgs = self.generator.predict(noise)

            # Discriminatorの学習
            # 二行になっているのは、ミニバッチの半分はFake,もう半分はRealであるから
            # このミニミニバッチを合わせて、学習するのは適切ではない
            d_loss_real = self.discriminator.train_on_batch(
                imgs, np.ones((half_batch, 1))
            )
            d_loss_fake = self.discriminator.train_on_batch(
                gen_imgs, np.zeros((half_batch, 1))
            )

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            """""" """""" """""" """""" """""" """""
            生成器の訓練(Training Generator)

            """ """""" """""" """""" """""" """""" ""

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))
            g_loss = self.combined.train_on_batch(
                noise, np.ones((batch_size, 1))
            )  # 生成器(Generater)の学習

            print(
                "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                % (epoch, d_loss[0], 100 * d_loss[1], g_loss)
            )

            model_dir = Path("ganmodels")
            model_dir.mkdir(exist_ok=True)
            if iteration % save_interval == 0:
                self.save_imgs(iteration, check_noise, r, c)
                start = np.expand_dims(check_noise[0], axis=0)
                end = np.expand_dims(check_noise[1], axis=0)
                resultImage = self.visualizeInterpolation(start=start, end=end)
                cv2.imwrite(
                    "images/latent/" + "latent_{}.png".format(iteration), resultImage
                )
                if iteration % model_interval == 0:
                    self.generator.save(
                        str(model_dir) + "/dcgan-{}-iter.h5".format(iteration)
                    )

    def save_imgs(self, iteration, check_noise, r, c):
        noise = check_noise
        gen_imgs = self.generator.predict(noise)

        # 0-1 rescale
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig("images/gen_imgs/kill_me_%d.png" % iteration)

        plt.close()

    def load_imgs(self):
        img_paths = []
        labels = []
        images = []
        for cl_name in self.class_names:
            img_names = os.listdir(os.path.join(root_dir, cl_name))
            for img_name in img_names:
                img_paths.append(
                    os.path.abspath(os.path.join(root_dir, cl_name, img_name))
                )
                hot_cl_name = self.get_class_one_hot(cl_name)
                labels.append(hot_cl_name)

        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        images = np.array(images)

        return (np.array(images), np.array(labels))

    def get_class_one_hot(self, class_str):
        label_encoded = self.class_names.index(class_str)

        label_hot = np_utils.to_categorical(label_encoded, len(self.class_names))
        label_hot = label_hot

        return label_hot

    def visualizeInterpolation(self, start, end, save=True, nbSteps=10):
        print("Generating interpolations...")

        steps = nbSteps
        latentStart = start
        latentEnd = end

        startImg = self.generator.predict(latentStart)
        endImg = self.generator.predict(latentEnd)

        vectors = []

        alphaValues = np.linspace(0, 1, steps)
        for alpha in alphaValues:
            vector = latentStart * (1 - alpha) + latentEnd * alpha
            vectors.append(vector)

        vectors = np.array(vectors)

        resultLatent = None
        resultImage = None

        for i, vec in enumerate(vectors):
            gen_img = np.squeeze(self.generator.predict(vec), axis=0)
            gen_img = (0.5 * gen_img + 0.5) * 255
            interpolatedImage = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
            interpolatedImage = interpolatedImage.astype(np.uint8)
            resultImage = (
                interpolatedImage
                if resultImage is None
                else np.hstack([resultImage, interpolatedImage])
            )

        return resultImage


if __name__ == "__main__":
    datarar = rar.RarFile("kill_me_baby_datasets.rar")
    datarar.extractall()

    dcgan = DCGAN()
    r, c = 5, 5
    check_noise = np.random.uniform(-1, 1, (r * c, 100))
    dcgan.train(check_noise=check_noise, r=r, c=c)
