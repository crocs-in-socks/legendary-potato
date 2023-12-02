import numpy as np
import matplotlib.pyplot as plt

data = np.load('./results/VGGproxy_classifier_weightedBCEpretrain_01_12_2023_accuracies.npy')
print(data.shape)
train_loss = data[0]
validation_loss = data[1]
train_accuracy = data[2]
validation_accuracy = data[3]

print(f'\tTrain loss\t\tValidation loss')
for idx, _ in enumerate(train_loss):
    print(f'{idx+1}\t{train_loss[idx]}\t{validation_loss[idx]}')

plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 1)
plt.plot(train_loss)
plt.plot(validation_loss)
plt.grid(True)
plt.title('Losses')
plt.subplot(1, 2, 2)
plt.plot(train_accuracy)
plt.plot(validation_accuracy)
plt.grid(True)
plt.title('Accuracies')
plt.savefig('./temp')
plt.close()
