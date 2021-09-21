import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import json

def load_model(filename):
    """
    从json文件里读取模型
    """
    dict = json.load(open(filename))
    mean_face = np.mat(dict['mean_face'])
    mean_face = np.reshape(mean_face, (10304,1))
    eigen_face = np.mat(dict['eigen_face'])
    eigen_face = np.reshape(eigen_face, (10304,-1))   
    diffTrain = np.mat(dict['diffTrain'])
    diffTrain = np.reshape(diffTrain, (10304,-1))
    
    return mean_face, eigen_face, diffTrain


def EigenFaceTesting(img_col, mean_face, eigen_face, diffTrain):
    """
    测试，返回训练集中找到的最相似图像
    """

    diffTest = img_col - mean_face              # 测试得到的偏差图像
    weightVec = eigen_face.T * diffTest         # 权重 110x1

    res = 0
    resVal = np.inf
    for i in range(np.shape(diffTrain)[1]):
        TrainVec = eigen_face.T * diffTrain[:,i]
        if  (np.array(weightVec - TrainVec)**2).sum() < resVal:
            res =  i
            resVal = (np.array(weightVec-TrainVec)**2).sum()
    res_img = 'att_faces/s' + str(res//5+1) + '/' + str(res%5+1) + '.pgm'
    return res_img


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ('Usage: python mytest.py ' \
            + '<test image dir> <import model path>')
        sys.exit(1)

    mean_face, eigen_face, diffTrain = load_model(str(sys.argv[2]))

    img = cv2.imread(str(sys.argv[1]), 0)
    img_col = np.array(img, dtype='float64').flatten()
    img_col = np.reshape(img_col, (10304,1))    # 目标人脸，10304x1

    test_res_img = EigenFaceTesting(img_col, mean_face, eigen_face, diffTrain)

    print(test_res_img)

    img1 = cv2.imread(str(sys.argv[1]))
    img2 = cv2.imread(test_res_img)
    
    alpha = 0.5
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)

    plt.figure()
    plt.suptitle('test result') 
    plt.subplot(1,3,1)
    plt.imshow(img1,cmap='gray'), plt.title('input'), plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(img2,cmap='gray'), plt.title('match'), plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(img_add,cmap='gray'), plt.title('mix'), plt.axis('off')
    plt.show()
