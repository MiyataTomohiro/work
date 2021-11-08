def build_discriminator(self):
    """
    build_discriminator() : 識別器(Discriminator)の処理(畳み込み)を行う関数
    入力 :  生成器(Generator)からの生成画像(fake)とオリジナル画像(real)の2種類
    処理 :  畳み込み
    詳細 :  全結合層を削除
            BatchNormalizationを頻繁に使用
            入力画像がfakeかrealを判別して分類
            中間層以外の活性化関数にLeakyReLU使用
            プーリング層の代わりにstride=2の畳み込み層を使用
    """

    # 入力される画像の形式を格納
    img_shape = self.shape

    # Sequential : モデル層を積み重ねる形式の記述方法(.addメソッドで簡単に層を追加可能)
    model = Sequential()

    """
    Conv2D(2次元の畳み込み層, 空間フィルタ畳み込み演算層)
    filters : 畳み込みにおける出力フィルタの数
    kernel_size : 2次元の畳み込みウィンドウの幅と高さを指定(単一の整数値の場合、正方形のカーネル)
    strides : 畳み込みの縦と横のスライドを指定(単一の整数の場合は幅と高さが同様)
    input_shape : モデルの最初のレイヤーは指定

    padding : 出力画像と入力画像の画像サイズの関係は"vaid"か"same"を指定
    "vaid"(size:出力画像>入力画像),"same"(size:出力画像=入力画像)

    Leaky ReLU(Leaky Rectified Linear Unit)関数
    入力値が0以下の場合 --> 出力値はα倍した値(α = alpha)
    入力値が0以上の場合 --> 出力値は入力値と同値

    Dropout : 過学習予防(Dropout(0.25)は全結合の層とのつながりを「25%」無効化)

    ZeroPadding2D : 画像のテンソルの上下左右にゼロの行と列を追加

    padding : タプル(2つの整数)のタプル -> 四辺それぞれにパディング ((top_pad, bottom_pad), (left_pad, right_pad))

    BatchNormalization : 各バッチごとに前の層の出力を正規化(平均を0、標準偏差を１に近づける変換)
    momentum : 移動平均のためのモーメント
    """

    # 畳み込み1層目
    model.add(
        Conv2D(
            filters=32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"
        )
    )
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # 畳み込み2層目
    model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    # 畳み込み3層目
    model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    # 畳み込み4層目
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    """
    Flatten : 平坦化(1次元ベクトルに変換)
    Dense : 通常の全結合ニューラルネットワークレイヤー
    units --> 1 : 出力空間の次元数
    活性化関数(activation) --> sigmoid(Standard sigmoid function)関数
    グラデーションで、0から1までの値を出力
    """

    # 出力層(今回は2値分類であるからシグモイド関数)
    model.add(Flatten())
    model.add(Dense(units=1, activation="sigmoid"))

    """
    model.summary() : モデルの要約を出力
    Input() : Kerasテンソルのインスタンス化に使用
    model() : 生成器の処理で作成したモデル
    Model() : テンソル(多次元配列)の入出力を与え, モデルをインスタンス化
    """

    model.summary()
    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)
