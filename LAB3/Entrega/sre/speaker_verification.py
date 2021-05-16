import pandas as pd
import torch
from metrics import EER, minDCF


def speaker_verification(path_dev_file, path_mean_file):
    df_dev = pd.read_csv(path_dev_file)
    df_mean = pd.read_csv(path_mean_file)

    embeddings_dev_list = df_dev.values.tolist()
    embeddings_mean_list = df_mean.values.tolist()

    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    trials = []
    target_scores = []
    non_target_scores = []
    for row_dev in embeddings_dev_list:
        for row_mean in embeddings_mean_list:
            file_dev = row_dev[0]
            id_dev = row_dev[1]
            id_mean = row_mean[0]
            emb1 = torch.FloatTensor(row_dev[4:])
            emb2 = torch.FloatTensor(row_mean[1:])
            score = similarity(emb1, emb2)
            # make trials (check if both have same id)
            if id_mean in file_dev:
                trials.append('\n' + id_mean + ' ' + file_dev + ' ' + 'TARGET')
                target_scores.append(score)
            else:
                trials.append('\n' + id_mean + ' ' + file_dev + ' ' + 'NONTARGET')
                non_target_scores.append(score)

    # write trials.txt file
    with open('trials.txt', 'w') as trials_file:
        trials_file.writelines(trials)

    # calculate metrics
    eer, eer_th = EER(torch.tensor(target_scores), torch.tensor(non_target_scores))
    min_dcf, min_dcf_th = minDCF(torch.tensor(target_scores), torch.tensor(non_target_scores))
    print('eer: ' + str(eer) + ', eer_th: ' + str(eer_th))
    print('min_dcf: ' + str(min_dcf) + ', min_dcf_th: ' + str(min_dcf_th))


if __name__ == '__main__':
    # select model: 'ecapa' or 'xvector'
    model = 'xvector'
    path_dev = 'voxceleb_pt/embeddings/devel_' + model + '.csv'
    path_mean = 'voxceleb_pt/embeddings/train_' + model + '_mean.csv'
    # speaker verification
    speaker_verification(path_dev, path_mean)
