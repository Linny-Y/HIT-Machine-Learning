import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image

def pca(x, k):
    """
        pca降维
    """
    mu = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    values, vectors = np.linalg.eig(cov)
    index = np.argsort(values)[: -(k + 1): -1]  # 取最大的 k 个的下标值
    vectors = vectors[:, index]  # 取对应下标的特征向量
    x_pca = (x - mu).dot(vectors).dot(vectors.T) + mu  # 重建数据
    return x_pca

def generate_data(mu, sigma, n=100):
    """
        生成高斯分布数据
    """
    x = np.random.multivariate_normal(mean=mu, cov=sigma, size=n)
    return x


def faces_pca(path, k_list):
    """
        path: 文件夹名称
        k_list: 不同维度列表
    """
    x_list = faces(path)
    for x in x_list:
        x_pca_list = []
        psnr_list = []
        for k in k_list:
            x_pca = pca(x, k)  
            x_pca_list.append(x_pca)
            psnr_list.append(psnr(x, x_pca))
        show_faces(x, x_pca_list, k_list, psnr_list)


def faces(path):
    """
        读取指定目录下的所有文件
    """
    file_list = os.listdir(path)
    x_list = []
    for file in file_list:
        file_path = os.path.join(path, file)
        pic = Image.open(file_path).convert("L")  # 读入图片 转换为灰度图
        x_list.append(np.asarray(pic))
    return x_list



def show_faces(x, x_pca_list, k_list, psnr_list):
    """
        展示降维后的结果
    """
    plt.figure(figsize=(12, 8), frameon=False)
    # 原图
    plt.subplot(3, 3, 1)
    plt.title("Original Picture")
    plt.imshow(x, cmap='gray')
    plt.axis("off")  # 去掉坐标轴
    # 降维后的图
    for i in range(len(k_list)):
        plt.subplot(3, 3, i + 2)
        plt.title(
            "k = " + str(k_list[i]) + ", PSNR = " + "{:.2f}".format(psnr_list[i])
        )
        plt.imshow(x_pca_list[i], cmap='gray')
        plt.axis("off")
    plt.show()
    

def psnr(source, target):
    """
        计算峰值信噪比
    """
    rmse = np.sqrt(np.mean((source - target) ** 2))
    return 20 * np.log10(255.0 / rmse)



if __name__ == '__main__':
    """
        三维到二维
    """
    mean = [1, 2, 3]
    cov = [[0.1, 0, 0], [0, 10, 0], [0, 0, 10]]
    x = generate_data(mean, cov)  # 默认100个数据
    # print(x)
    x_pca = pca(x, 2)
    # print(x_pca)
    plt.style.use("seaborn")
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="cornflowerblue", label="Origin Data")
    ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c="crimson", label="PCA Data")
    ax.plot_trisurf(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], color="r", alpha=0.3)
    ax.legend(loc="best")
    plt.show()
    plt.style.use("default")

    """
        人脸数据
    """
    k_list = [50, 30, 20, 10, 5, 4, 2, 1]
    faces_pca('faces', k_list)