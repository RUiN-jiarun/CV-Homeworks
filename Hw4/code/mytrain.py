import cv2
import numpy as np
import sys
import json
from matplotlib import pyplot as plt
import os

np.set_printoptions(threshold=np.inf)

def import_data_from_orl(filename):
    """
    从数据集读取数据
    return:
    train_images: 包含训练数据的列表，每一项为numpy数组
    test_images: 包含测试数据的列表，每一项为numpy数组
    """
    train_images = []
    test_images = []
    for face_id in range(1, 42):
        for training_id in range(1, 6):
            path_to_img = os.path.join(filename,
                    's' + str(face_id), str(training_id) + '.pgm')         
            img = cv2.imread(path_to_img, 0)                               
            img_col = np.array(img, dtype='float64').flatten()             
            train_images.append(img_col)
        for testing_id in range(6, 11):
            path_to_img = os.path.join(filename,
                    's' + str(face_id), str(training_id) + '.pgm')        
            img = cv2.imread(path_to_img, 0)                               
            img_col = np.array(img, dtype='float64').flatten()            
            test_images.append(img_col)
           
    return train_images, test_images


def EigenFaceTraining(FaceMat, energy=0.8):  
    """
    进行数据训练
    Param: 
    FaceMat: 全数据矩阵(205x10304)
    energy: 能量 default=0.8
    """
    FaceMat = np.mat(FaceMat).T         # 全数据矩阵转置(10304x205)
    mean_face = np.mean(FaceMat, 1)     # 平均脸 
    diffTrain = FaceMat - mean_face     # 得到插值图像的数据矩阵

    '''
    covMat = np.mat(diffTrain * diffTrain.T)    # 协方差矩阵(10304x10304)
    eigvals, eigVects = np.linalg.eig(np.mat(covMat))   # 求协方差矩阵的特征值和特征向量(10304x10304)
    print(eigVects.shape)
    这太慢了！！！
    '''
    eigvals, eigVects = np.linalg.eig(np.mat(diffTrain.T * diffTrain))  # 205个特征向量
    eigSortIndex = np.argsort(-eigvals) # 特征值排序
    for i in range(np.shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= energy: # 按energy参数保留
            eigSortIndex = eigSortIndex[:i]
            break
    eigen_face = diffTrain * eigVects[:, eigSortIndex]    # 得到特征脸(在0.85energy下是10304x47)
    
    return mean_face, eigen_face, diffTrain


def export_model(mean_face, eigen_face, diffTrain, filename):
    """
    保存model到目标json文件
    """
    model = {'mean_face':mean_face.__str__(), 'eigen_face':eigen_face.__str__(), 'diffTrain':diffTrain.__str__()}
    json.dump(model, open(filename,'w'))


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print ('Usage: python mytrain.py ' \
            + '<att faces dir> ' + '<export model path> [<energy>]')
        sys.exit(1)

    train_images, test_images = import_data_from_orl(str(sys.argv[1]))
    if len(sys.argv) == 3:   
        mean_face, eigen_face, diffTrain = EigenFaceTraining(train_images)
    
    if len(sys.argv) == 4:                                                 
        mean_face, eigen_face, diffTrain = EigenFaceTraining(train_images, float(sys.argv[3]))

    # export_model(mean_face, eigen_face, diffTrain, str(sys.argv[2]))
    
    print('PC number = ', eigen_face.shape[1])      # 0.95 -- 110

    # 显示平均脸
    img = np.reshape(mean_face, (112, 92))
    img = img.astype(np.uint8)

    # 显示前十张特征脸
    img_eigen = []
    eigens = eigen_face[:,0:10]
    for i in range(10):
        eigen = eigens[:,i]
        eigen = 255*(eigen-np.min(eigen))/(np.max(eigen)-np.min(eigen))
        eigen = np.reshape(eigen, (112, 92))
        img_eigen.append(eigen.astype(np.uint8))

    # 特征脸的叠加
    eigen_mix = eigen_face.sum(axis=1)
    img_mix = np.reshape(eigen_mix, (112, 92))
    img_mix = img_mix.astype(np.uint8)

    alpha = 0.5
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img_eigen[0], alpha, img_eigen[1], beta, gamma)
    for i in range(2,10):
        img_add = cv2.addWeighted(img_add, alpha, img_eigen[i], beta, gamma)

    plt.title('mean face') 
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.suptitle('eigen faces') 
    plt.subplot(2,5,1)
    plt.imshow(img_eigen[0],cmap='gray'), plt.axis('off')
    plt.subplot(2,5,2)
    plt.imshow(img_eigen[1],cmap='gray'), plt.axis('off') 
    plt.subplot(2,5,3)
    plt.imshow(img_eigen[2],cmap='gray'), plt.axis('off')
    plt.subplot(2,5,4)
    plt.imshow(img_eigen[3],cmap='gray'), plt.axis('off')
    plt.subplot(2,5,5)
    plt.imshow(img_eigen[4],cmap='gray'), plt.axis('off')
    plt.subplot(2,5,6)
    plt.imshow(img_eigen[5],cmap='gray'), plt.axis('off')
    plt.subplot(2,5,7)
    plt.imshow(img_eigen[6],cmap='gray'), plt.axis('off')
    plt.subplot(2,5,8)
    plt.imshow(img_eigen[7],cmap='gray'), plt.axis('off')
    plt.subplot(2,5,9)
    plt.imshow(img_eigen[8],cmap='gray'), plt.axis('off')
    plt.subplot(2,5,10)
    plt.imshow(img_eigen[9],cmap='gray'), plt.axis('off')
    plt.show()
    
    plt.title('eigen faces mixed') 
    plt.imshow(img_add, cmap='gray')
    plt.axis('off')
    plt.show()