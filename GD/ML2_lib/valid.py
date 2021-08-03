import numpy as np


# n_blockに分ける、seqは入力データ
def median_of_means(seq, n_blocks):
    # 分割数がデータの長さより大きいとだめ
    if n_blocks > len(seq):
        n_blocks = int(np.ceil(len(seq) / 2))
    # indexでシャッフル
    indic = np.array(list(range(n_blocks)) * int(len(seq) / n_blocks))
    np.random.shuffle(indic)
    # グループごとの平均
    means = [np.mean(seq[list(np.where(indic == block)[0])]) for block in range(n_blocks)]
    # 中央値を返す
    return np.median(means)

def median_of_means_by_torch(seq,n_blocks):
    if n_blocks > len(seq):
        n_blocks = int(np.ceil(len(seq) / 2))


