from jiwer import wer

original = "vamos nós todos contentes futuro em caraças entrar no carro seguimos temos no carro uma auto estrada nem é esse porque a criança que lê sizwe sai do aeroporto de milão auto-estrada nenhum quilômetro na mas nenhum km treinando tudo que eu estou conta lembrando não andarmos 1 km"
hypothesis = "vamos nós todos contentes para Tourin caraças entrámos no carro, seguimos metemo-nos no carro. Pumba, autoestrada. Nem, é assim, para quem conhece, sai do aeroporto de Milão, autoestrada. Nem 1 km andámos. Nem 1 km andámos. Tudo o que eu vos estou a contar é verdade. Não andámos 1 km"

error = wer(original, hypothesis)

print(error)