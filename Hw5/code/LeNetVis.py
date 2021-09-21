from matplotlib import pyplot as plt

epoch = range(1,11)
avgloss = [0.2001, 0.0982, 0.0718, 0.0616, 0.0590, 0.0436, 0.0403, 0.0442, 0.0399, 0.0394]
corr = [9395,9687,9773,9795,9804,9852,9876,9862,9870,9871]
accu = [i/10000.0000 for i in corr]
print(accu)

plt.figure()
plt.subplot(2,1,1)
plt.plot(epoch, avgloss)
plt.xticks(epoch,epoch[::1])
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.subplot(2,1,2)
plt.plot(epoch,accu)
plt.xticks(epoch,epoch[::1])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()