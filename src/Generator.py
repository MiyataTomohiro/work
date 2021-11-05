def build_generator(self):
    """
　　build_generator() : 生成器(Generator)の処理(転置畳み込み)を行う関数
　　入力 :  128×128×3の画像
　　処理 :  転置畳み込み
　　詳細 :  最後の畳み込みフィルタが3枚(画像のチャネル数と一致)
　　        はじめのノード数は32×32×128(転置畳み込みを2回行うから)
　　        32×32 --> 64×64 --> 128×128と画像が変化するように設定(Conv2Dのpadding='same'に注意)
　　"""
    
    # 一様乱数や正規分布から抽出した潜在変数(nosie)の次元数を指定
    nosie_shape = (self.z_dim,)

    # Sequential : モデル層を積み重ねる形式の記述方法(.addメソッドで簡単に層を追加可能)
    model = Sequential()

    """"
    Dense : 通常の全結合ニューラルネットワークレイヤー
    units --> 128*32*32 : 出力空間の次元数
    activation(活性化関数) --> relu(Rectified Linear Unit)関数
    入力値が0以下の場合 --> 出力値が常に0, 入力値が0以上の場合 --> 出力値が入力値と同値

    input_shape --> noise : モデルの最初のレイヤーは指定

    Reshape --> 32*32*128 : 指定したサイズ(target_shape)に出力を変形

    BatchNormalization : 各バッチごとに前の層の出力を正規化(平均を0、標準偏差を１に近づける変換)
    momentum : 移動平均のためのモーメント
    UpSampling2D : 2次元の入力に対するアップサンプリングレイヤー
    """

    model.add(Dense(units=128 * 32 * 32, activation="relu", input_shape=noise))
    model.add(Reshape(target_shape=(32, 32, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())

    """
    Conv2D(2次元の畳み込み層, 空間フィルタ畳み込み演算層)
    filters : 畳み込みにおける出力フィルタの数
    kernel_size : 2次元の畳み込みウィンドウの幅と高さを指定(単一の整数値の場合、正方形のカーネル)

    padding : 出力画像と入力画像の画像サイズの関係は"vaid"か"same"を指定
    "vaid"(size:出力画像>入力画像),"same"(size:出力画像=入力画像)

    BatchNormalization : 各バッチごとに前の層の出力を正規化(平均を0、標準偏差を１に近づける変換)
    momentum : 移動平均のためのモーメント

    活性化関数(Activation)
    relU関数 : 入力値が0以下の場合 -->出力値が常に0, 入力値が0以上の場合 --> 出力値が入力値と同値
    tanh関数 : あらゆる入力値を-1.0～1.0の範囲の数値に変換して出力
    """

    # 転置畳み込み1回目
    model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())

    # 転置畳み込み2回目
    model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    # 画像のチャンネル数=3と一致させる処理
    model.add(Conv2D(filters=3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    """
    model.summary() : モデルの要約を出力
    Input() : Kerasテンソルのインスタンス化に使用
    model() : 生成器の処理で作成したモデル
    Model() : テンソルの入出力を与え, モデルをインスタンス化
    """

    model.summary()
    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)
