# 画像処理モジュール "Pillow" で画像をリサイズする。
import glob
import os

from PIL import Image


# jpg形式ファイルの画像サイズを変更する
def resizeImage(inputImage, outputImage, filename, num):
    # 元画像読み込み
    img = Image.open(inputImage)
    # リサイズ
    image = img.resize(size=(128, 128), resample=Image.LANCZOS)
    outputPath = os.path.join(outputImage, f"{name}_{num:0>3}.jpg")
    # 画像の保存
    image.save(outputPath, quality=100)


# 入出力パスの設定
name = input("画像フォルダー名を入力>>")
output_path = os.path.join(f"/home/miyata/test/img/input/{name}/")
if not os.path.exists(output_path): #ディレクトリがなかったら
    os.mkdir(output_path)   #作成したいフォルダ名を作成
n = 0

img_path = glob.glob(f"/home/miyata/test/img/download/{name}/*.jpg")

for img in img_path:
    n += 1
    resizeImage(img, output_path, name, n)
