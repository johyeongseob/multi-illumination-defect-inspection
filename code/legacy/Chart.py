import matplotlib.pyplot as plt


# Before

# model2_BC
# train_acc = [0.68, 0.74, 0.79, 0.83, 0.86, 0.86, 0.87, 0.90, 0.94, 0.94]
# valid_acc = [0.68, 0.71, 0.49, 0.69, 0.70, 0.69, 0.70, 0.72, 0.72, 0.71]
# test_acc  = [0.69, 0.69, 0.65, 0.66, 0.65, 0.66, 0.70, 0.66, 0.69, 0.69]

# model2_MC
# train_acc = [0.62, 0.71, 0.79, 0.83, 0.85, 0.90, 0.94, 0.94, 0.92, 0.93]
# valid_acc = [0.55, 0.57, 0.61, 0.56, 0.60, 0.59, 0.60, 0.57, 0.60, 0.61]
# test_acc  = [0.57, 0.55, 0.59, 0.60, 0.60, 0.60, 0.58, 0.59, 0.58, 0.61]

# model3
train_acc = [0.68, 0.76, 0.81, 0.87, 0.91, 0.93, 0.97, 0.97, 0.96, 0.98]
valid_acc = [0.72, 0.74, 0.74, 0.73, 0.72, 0.68, 0.74, 0.77, 0.76, 0.74]
test_acc  = [0.69, 0.72, 0.71, 0.73, 0.75, 0.70, 0.71, 0.74, 0.70, 0.69]


# After

# model2_BC
# pretrain_train_acc = [0.62, 0.73, 0.82, 0.89, 0.92, 0.94, 0.94, 0.94, 0.94, 0.97]
# pretrain_valid_acc = [0.68, 0.63, 0.62, 0.67, 0.64, 0.65, 0.65, 0.66, 0.68, 0.69]
# pretrain_test_acc  = [0.68, 0.67, 0.68, 0.69, 0.69, 0.67, 0.68, 0.68, 0.68, 0.64]


# model2_MC
# pretrain_train_acc = [0.54, 0.68, 0.76, 0.83, 0.88, 0.92, 0.96, 0.99]
# pretrain_valid_acc = [0.49, 0.54, 0.56, 0.59, 0.64, 0.56, 0.56, 0.56]
# pretrain_test_acc  = [0.49, 0.53, 0.60, 0.63, 0.64, 0.62, 0.62, 0.62]


# model3
pretrain_train_acc = [0.61, 0.77, 0.82, 0.91, 0.94, 0.98]
pretrain_valid_acc = [0.67, 0.75, 0.78, 0.82, 0.79, 0.73]
pretrain_test_acc  = [0.65, 0.67, 0.67, 0.73, 0.71, 0.70]

plt.figure(figsize=(20, 12))

x = range(1, len(train_acc) + 1)
x_ = range(1, len(pretrain_train_acc) + 1)

plt.plot(x, train_acc, color='r', label='Train Acc', linestyle='--', marker='o')
plt.plot(x, valid_acc, color='r', label='Valid Acc', linestyle='--', marker='s')
plt.plot(x, test_acc,  color='r', label='Test Acc', linestyle='--', marker='^')

plt.plot(x_, pretrain_train_acc, color='b', label='Pretrain_train Acc', linestyle='--', marker='o')
plt.plot(x_, pretrain_valid_acc, color='b', label='Pretrain_valid Acc', linestyle='--', marker='s')
plt.plot(x_, pretrain_test_acc,  color='b', label='Pretrain_test Acc', linestyle='--', marker='^')

plt.title('Model3_accuracy Chart')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()