import numpy as np
import matplotlib.pyplot as plt

# data_first = np.load('./results/VGGproxy_classifier_weightedBCEpretrain_batch6_02_12_2023_accuracies.npy')
# data_second = np.load('./results/VGGproxy_classifier_weightedBCEpretrain_batch6_after28epoch_02_12_2023_accuracies.npy')
# print(data_first.shape, data_second.shape)
# train_loss = np.concatenate([data_first[0, :28], data_second[0]])
# validation_loss = np.concatenate([data_first[1, :28], data_second[1]])
# train_accuracy = np.concatenate([data_first[2, :28], data_second[2]])
# validation_accuracy = np.concatenate([data_first[3, :28], data_second[3]])

file_name = 'LiTS_Unet_preprocessing_0_>_400_window_init_features_64_with_crop_cv2unit8_median_filtering_and_bilateral:5050_and_AHE_cl:1_02_02_2024_losses'
data = np.load(f'./results/{file_name}.npy')
print(data.shape)
train_loss = data[0]
validation_loss = data[2]
train_accuracy = data[1]
validation_accuracy = data[3]

print(f'\tTrain loss\t\tValidation loss\t\tTrain accuracy\t\tValidation accuracy')
for idx, _ in enumerate(train_loss):
    print(f'{idx+1}\t{train_loss[idx]}\t{validation_loss[idx]}\t{train_accuracy[idx]}\t{validation_accuracy[idx]}')

print(f'\nMinimum loss at epoch #{np.argmin(validation_loss)}, loss was : {np.min(validation_loss)}, accuracy at this loss: {validation_accuracy[np.argmin(validation_loss)]}')
print(f'Maximum accuracy: {validation_accuracy[np.argmax(validation_accuracy)]} at epoch #{np.argmax(validation_accuracy)}')

plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 1)
plt.plot(train_loss)
plt.plot(validation_loss)
plt.grid(True)
plt.title('Classification Losses')
plt.subplot(1, 2, 2)
plt.plot(train_accuracy)
plt.plot(validation_accuracy)
plt.grid(True)
plt.title('Classification Accuracies')
plt.savefig(f'./temp')
plt.close()
