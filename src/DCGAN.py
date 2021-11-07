




class DCGAN:
    """
    self :  クラス構造を取る際の定型の構文(インスタンス自身, その時点の自分, メソッドの仮引数)
    init :  コンストラクタと呼ばれる初期化のための関数(メソッド)
            インスタンス化を行う時に必ず最初に呼び出される特殊な関数(メソッド)
            オブジェクト生成(インスタンスを生成)のついでにデータの初期化を行うもの
    """
    def __init__(self):
        # クラス分類用のクラス名
        self.class_names = os.listdir(root_dir)

        # 入力画像サイズ(shape)と潜在変数の次元(z_dim)
        self.shape = (128, 128, 3)
        self.z_dim = 100

        """
        optimizer : オプティマイザー(最適化アルゴリズム)
        Adam  --> lr : 学習率(0以上の浮動小数点数)
                  beta_1 : 一般的に1に近い値(0<beta<1の浮動小数点数)
        """
        optimizer = Adam(lr=0.0002, beta_1=0.5)

        """
        識別器(Discriminator)はRealかfakeを見分ける二値分類を行うためbinary_crossentropyを使用
        
        クロスエントロピー : 一種の距離を表す指標(2つの値がどれだけ離れているかを示す尺度)
        binary_crossentropy : バイナリクロスエントロピー
        [0,1]をとる変数と2クラスラベルにベルヌーイ分布を仮定した場合の対数尤度

        compile(コンパイル) : モデルの学習を始める前に、どのような学習処理を実行するか設定
        loss(損失関数) : 　モデルが最小化しようとする目的関数
        optimizer(最適化アルゴリズム) : Optimizerクラスのインスタンスを与える
        metrics(評価関数のリスト) : 分類問題では精度としてmetrics=['accuracy']を指定
        """
        self.discriminator = self.build_discriminator()
        self.discrinator.compile(
            loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']
        )
        
        """
        生成器(Generator)学習用の組合せモデル(Combined_model)を作成
        Input() : Kerasテンソルのインスタンス化に使用

        識別器(Discriminator)のパラメータは固定
        --> .trainable = False のように設定するとパラメータが学習中に更新されない
        """
        self.generator = self.build_generator()
        z = Input(shape=(100,)) # 入力値(z) : 潜在変数(Noise)
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img) # 出力値(Valid): Real(=1), Fake(=0)を取る

        """
        Model() : テンソルの入出力を与え, モデルをインスタンス化
        compile(コンパイル) : モデルの学習を始める前に、どのような学習処理を実行するか設定
        loss(損失関数) : 　モデルが最小化しようとする目的関数
        --> combined_model(Generatorの学習用モデル)も二値分類を行うためbinary_crossentropy
        optimizer(最適化アルゴリズム) : Optimizerクラスのインスタンスを与える
        """
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    
    def build_combined(self):
        """
        build_combined() : 生成器(Generator)学習用の組合せモデル(Combined_model)を作成する関数

        識別器(Discriminator)のパラメータは固定
        --> .trainable = False のように設定するとパラメータが学習中に更新されない

        Sequential() : モデル層を積み重ねる形式の記述方法
        """
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])

    def train(self, epochs, batch_size=128, save_interval=50):
        X_train, labels = self.load_imgs()

        half_batch = int(batch_size / 2)

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        for epoch in range(epochs):

            # ------------------
            # Training Discriminator
            # -----------------
            idx = np.random.randint(0, X_train.shape[0], half_batch)

            imgs = X_train[idx]

            noise = np.random.uniform(-1, 1, (half_batch, self.z_dim))

            gen_imgs = self.generator.predict(noise)

            # Discriminatorの学習
            # 二行になっているのは、ミニバッチの半分はFake,もう半分はRealであるから
            # このミニミニバッチを合わせて、学習するのは適切ではない
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            # Training Generator
            # -----------------

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))

            # Generaterの学習
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))



