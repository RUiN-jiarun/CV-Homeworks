from mytrain import import_data_from_orl, EigenFaceTraining
from mytest import EigenFaceTesting
import cv2
import numpy as np
import os 
from matplotlib import pyplot as plt

if __name__ == '__main__':
    train_images, t = import_data_from_orl('att_faces')

    acc = []
    pcs = []

    for i in np.arange(0,1,0.05):
        print(i)
        mean_face, eigen_face, diffTrain = EigenFaceTraining(train_images, 0.8)

        accu_num = 0
        
        for face_id in range(1, 42):
            for i in range(6, 11):
                path_to_img = os.path.join('att_faces',
                        's' + str(face_id), str(i) + '.pgm')     
                img = cv2.imread(path_to_img, 0)                               
                img_col = np.array(img, dtype='float64').flatten()
                img_col = np.reshape(img_col, (10304,-1))           
                res = EigenFaceTesting(img_col, mean_face, eigen_face, diffTrain)
                num = int(res[res.find('s',10)+1: res.find('/',12)])
                if face_id == num:
                    accu_num += 1
        print(accu_num)
        accuracy = accu_num / 205
        print(accuracy)
        acc.append(accuracy)
        print(eigen_face.shape[1])
        pcs.append(eigen_face.shape[1])

    # with open('res.txt', 'w') as f:
    #     f.write(str(acc))
    #     f.write(str(pcs))


    # acc = [0.024390243902439025, 0.16585365853658537, 0.16585365853658537, 0.16585365853658537, 0.16585365853658537, 0.35609756097560974, 0.35609756097560974, 0.5756097560975609, 0.6146341463414634, 0.6536585365853659, 0.6926829268292682, 0.7317073170731707, 0.7414634146341463, 0.7707317073170732, 0.7658536585365854, 0.7853658536585366, 0.7853658536585366, 0.7853658536585366, 0.7853658536585366, 0.7853658536585366]
    x_ticks = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    # pcs = [0, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 9, 12, 16, 22, 32, 46, 69, 110]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(pcs, acc, 'r', lw=1, ms=5)
    plt.yticks(np.linspace(0,1,20))
    plt.ylabel('Accuracy')
    plt.xlabel('PCs')
    plt.title('Accuracy-PCs')
    plt.subplot(2,1,2)
    plt.plot(x_ticks, acc, 'g', lw=1, ms=5)
    plt.yticks(np.linspace(0,1,20))
    plt.ylabel('Accuracy')
    plt.xlabel('energy')
    plt.title('Accuracy-energy')
    plt.show()

