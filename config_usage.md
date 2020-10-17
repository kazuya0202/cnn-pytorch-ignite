# Usage: user_config

## 注意点

+ 変更してはいけないもの

  > + インデント（字下げ）
  > + パラメータ名
  > + `path`、`dataset`... の入れ子の親（× ~~コメントアウト~~、~~消す~~）

+ ファイルパスを指定するときは、`\\`か`/`を使う。

+ `default value`から変更しない場合、コメントアウト or 消してもOK。
  
  + コメントアウトするには、行の先頭に`#`をつける。

<br>

## path

| key          | type  | default value | description                          |
| ------------ | ----- | ------------- | ------------------------------------ |
| `dataset`    | `str` | `./dataset`   | データセットのパス [[^1]](#1)        |
| `result_dir` | `str` | `./results`   | 学習結果を保存するトップディレクトリ |

<br>

## dataset

| key                    | type            | default value            | description                                                |
| ---------------------- | --------------- | ------------------------ | ---------------------------------------------------------- |
| `limit_size`           | `int`           | `-1`                     | 各データセットで使用する画像のの最大枚数（`-1`は制限なし） |
| `test_size`            | `[int / float]` | `0.1`                    | テスト画像の割合・枚数（小数→割合 / 整数→枚数）            |
| `extensions`           | `List[str]`     | `["jpg", "png", "jpeg"]` | 対象画像の拡張子                                           |
| `is_shuffle_per_epoch` | `bool`          | `true`                   | エポック毎にデータセットをシャッフルする                   |
| `is_pre_splited`       | `bool`          | `false`                  | 学習用とテスト用に分けている                               |
| `train_dir`            | `str`           | `"./dataset/train"`      | 学習用画像のディレクトリ [[^1]](#1)                        |
| `valid_dir`            | `str`           | `"./dataset/valid"`      | テスト用画像のディレクトリ [[^1]](#1)                      |

<br>

## gradcam

| key             | type   | default value | description                        |
| --------------- | ------ | ------------- | ---------------------------------- |
| `enabled`       | `bool` | `false`       | Grad-CAMを実行する                 |
| `only_mistaken` | `bool` | `true`        | 間違えたときだけGrad-CAMを実行する |
| `layer`         | `str`  | `conv5`       | 可視化する層                       |

<br>

## network

| key                   | type   | default value | description                                  |
| --------------------- | ------ | ------------- | -------------------------------------------- |
| `height`              | `int`  | `60`          | 画像の入力サイズ（高さ）                     |
| `width`               | `int`  | `60`          | 画像の入力サイズ（幅）                       |
| `channels`            | `int`  | `3`           | 画像のチャンネル数 [[^2]](#2)                |
| `epoch`               | `int`  | `10`          | エポック数                                   |
| `batch`               | `int`  | `128`         | 1バッチの画像枚数（何枚ずつ行うか）          |
| `subdivision`         | `int`  | `4`           | バッチの細分化                               |
| `save_cycle`          | `int`  | `0`           | 学習モデルの保存サイクル [[^3]](#3)          |
| `test_cycle`          | `int`  | `1`           | 学習モデルのテストサイクル [[^3]](#3)        |
| `gpu_enabled`         | `bool` | `true`        | GPUを使用する（GPUが使用できない場合は無視） |
| `is_save_final_model` | `bool` | `true`        | 最終モデルを保存する                         |
| `class_name`          | `str`  | `"Net"`       | ネットワークの名前                           |
| `optim_name`          | `str`  | `"Adam"`      | オプティマイザの名前                         |

<br>

## option

| key                          | type   | default value | description                                      |
| ---------------------------- | ------ | ------------- | ------------------------------------------------ |
| `is_show_network_difinition` | `bool` | `true`        | 構築したネットワーク、クラスなどの定義を表示する |
| `is_save_log`                | `bool` | `true`        | 標準出力をファイルに保存する                     |
| `is_save_mistaken_pred`      | `bool` | `false`       | 学習中に間違えた画像を保存                       |
| `is_save_config`             | `bool` | `false`       | 構成を保存 [[^4]](#4)                            |
| `log_tensorboard`            | `bool` | `false`       | TensorBoardに保存                                |
| `is_save_cm`                 | `bool` | `false`       | 混合行列（Confusion Matrix）を保存               |

<br>

## Notes

#### [^1]

データセットのパスは、各クラスごとにフォルダにまとめ、その親フォルダのパスを指定する。

> 例
>
> ```
> Images/                <- ここを指定
> ├─ class_A/
> │      ├─ 001.jpg
> │      └─ ...
> ├─ class_B/
> │      ├─ 001.jpg
> │      └─ ...
> └─ ...
> ```

#### [^2]

`3`：カラー画像(RGB)　｜　`1`グレースケール画像

#### [^3]

`0`：何もしない　｜　`10`：10エポック毎に実行　｜　`N`：Nエポック毎に実行

#### [^4]

クラス，（乱数で分けたときの）学習用・テスト用画像のパス

