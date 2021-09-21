import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
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
    # diffTrain = np.mat(dict['diffTrain'])
    # diffTrain = np.reshape(diffTrain, (10304,-1))

    print(eigen_face.shape)
    
    return mean_face, eigen_face

"""TODO: passing value debug"""
def reconstruct(filename, mean_face, eigen_face):
    """
    将输入图像变换到特征脸空间，进行重建
    """
    avg = np.reshape(mean_face, (112, 92)).copy()
    output = np.reshape(mean_face, (112, 92)).copy()          # 从平均脸开始重建

    output1 = np.zeros((112,92))
    output2 = np.zeros((112,92))
    output3 = np.zeros((112,92))
    output4 = np.zeros((112,92))
    output5 = np.zeros((112,92))
    output6 = np.zeros((112,92))    
    img = cv2.imread(filename, 0)
    img_col = np.array(img, dtype='float64').flatten()
    img_col = np.reshape(img_col, (10304,1))
    diffTest = img_col - mean_face                  # 差异图像
    # weightVec = eigen_face.T * diffTest   
    # print(np.linalg.norm(eigen_face, axis=0) )   
    eigen_face = eigen_face / np.linalg.norm(eigen_face, axis=0) 
    
    # 重构人脸的时候要做一下这一步！！      

    for i in range(eigen_face.shape[1]):     
        weight = diffTest.T * eigen_face[:,i] 
        # print(weight.shape)
        output = output + float(weight) * eigen_face[:,i].reshape(112,92)
        # print(float(weight))
        if i == 9:
            output1 = output.copy()
        elif i == 24:
            output2 = output.copy()
        elif i == 49:
            output3 = output.copy()
        elif i == 99:
            output4 = output.copy()
        elif i == 199:
            output5 = output.copy()
        elif i == 299:
            output6 = output.copy()

    return img, avg, output1, output2, output3, output4, output5, output6


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ('Usage: python myreconstruct.py ' \
            + '<reconstruct image dir> <import model path>')
        sys.exit(1)

    mean_face, eigen_face = load_model(str(sys.argv[2]))

    img, avg, output1, output2, output3, output4, output5, output6 = reconstruct(str(sys.argv[1]), mean_face, eigen_face)

    plt.figure()
    plt.suptitle('reconstruct result') 
    plt.subplot(2,4,1)
    plt.imshow(img,cmap='gray'), plt.title('input'), plt.axis('off')
    plt.subplot(2,4,2)
    plt.imshow(avg,cmap='gray'), plt.title('mean face'), plt.axis('off')
    plt.subplot(2,4,3)
    plt.imshow(output1,cmap='gray'), plt.title('10PCs'), plt.axis('off')
    plt.subplot(2,4,4)
    plt.imshow(output2,cmap='gray'), plt.title('25PCs'), plt.axis('off')
    plt.subplot(2,4,5)
    plt.imshow(output3,cmap='gray'), plt.title('50PCs'), plt.axis('off')
    plt.subplot(2,4,6)
    plt.imshow(output4,cmap='gray'), plt.title('100PCs'), plt.axis('off')
    plt.subplot(2,4,7)
    plt.imshow(output5,cmap='gray'), plt.title('200PCs'), plt.axis('off')
    plt.subplot(2,4,8)
    plt.imshow(output5,cmap='gray'), plt.title('300PCs'), plt.axis('off')
    plt.show()