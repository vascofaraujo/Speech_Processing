import pandas as pd
import torch


def speaker_identification(path_test_file, path_mean_file):
    df_test = pd.read_csv(path_test_file)
    df_mean = pd.read_csv(path_mean_file)

    embeddings_test_list = df_test.values.tolist()
    embeddings_mean_list = df_mean.values.tolist()

    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    test_ident = []
    for row_test in embeddings_test_list:
        highest_score = 0
        highest_score_speaker = ''
        for row_mean in embeddings_mean_list:
            file_test = row_test[0]
            id_test = row_test[1]
            id_mean = row_mean[0]
            emb1 = torch.FloatTensor(row_test[4:])
            emb2 = torch.FloatTensor(row_mean[1:])
            score = similarity(emb1, emb2)
            # best score approach
            if score > highest_score:
                highest_score = score
                highest_score_speaker = id_mean
        test_ident.append('\n' + file_test + ' ' + highest_score_speaker)

    # write test_ident.txt file
    with open('test_ident.txt', 'w') as test_ident_file:
        test_ident_file.writelines(test_ident)


if __name__ == '__main__':
    # select model: 'ecapa' or 'xvector'
    model = 'ecapa'
    path_mean = 'voxceleb_pt/embeddings/train_' + model + '_mean.csv'
    path_test = 'voxceleb_pt/embeddings/test_' + model + '.csv'
    # speaker identification
    speaker_identification(path_test, path_mean)
