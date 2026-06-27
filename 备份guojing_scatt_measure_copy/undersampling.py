import numpy as np
import pandas as pd


def undersample_organoids(DF, surface_col='Organoids_Surface', random_seed=None):
    """
    基于表面积的欠采样方法实现
    参数：
        DF: 包含类器官数据的DataFrame
        surface_col: 表面积列的名字
        random_seed: 随机种子(保证可重复性)
    返回：
        降采样后的DataFrame
    """

    # 设置随机种子
    np.random.seed(random_seed)

    # 计算等效直径(D_surface = sqrt(S/pi))
    DF['equivalent_diameter'] = np.sqrt(DF[surface_col] / np.pi)

    # 创建采样掩码
    mask_small = DF['equivalent_diameter'] <= 10
    mask_medium = (DF['equivalent_diameter'] >10) & (DF['equivalent_diameter'] <= 20)

    # 对各区间进行采样
    # 小尺寸类器官采样率 0.01
    DF_small = DF[mask_small].sample(frac=0.01, random_state=random_seed)

    # 中等尺寸类器官采样率 0.1
    DF_medium = DF[mask_medium].sample(frac=0.1, random_state=random_seed)

    # 大尺寸类器官保留全部
    DF_large = DF[~mask_small & ~mask_medium]

    # 合并所有采样后的数据
    undersampled_DF = pd.concat([DF_small, DF_medium, DF_large])

    # 重置索引
    undersampled_DF = undersampled_DF.reset_index(drop=True)

    return undersampled_DF,  DF['equivalent_diameter'] 


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    np.random.seed(42)
    # # Path -> Excel表格路径
    # Path_1 = 'Data/FXN/20230701/excel'
    # Path_2 = 'Data/FXN/20230703/excel'
    # SavePath = 'Data/FXN'
    #
    # file_ls1 = os.listdir(Path_1)
    # file_ls2 = os.listdir(Path_2)
    # ROOTS_Start = [os.path.join(Path_1, file) for file in file_ls1]
    # ROOTS_End = [os.path.join(Path_2, file) for file in file_ls2]
    #
    # # 数据组合
    # ROOTS = ROOTS_Start + ROOTS_End
    # Data_All = []
    # for ROOT in ROOTS:
    #     DF = pd.read_excel(ROOT, sheet_name='Sheet1')
    #     Data_All.append(DF)
    # Data_All = pd.concat(Data_All,ignore_index=True)

    path = 'E:\student\Private\student11\Measure\Measure_copy\Data\ICC005_20240424\measure\ICC005_0424_ALL.xlsx'
    Data_All = pd.read_excel(path, sheet_name='Sheet1')

    # 表面积欠采样
    undersampled_df, oringin_equi_diameter = undersample_organoids(Data_All, surface_col='Organoids_Surface', random_seed=42)
    test1_1 = oringin_equi_diameter
    test1_2 = undersampled_df['equivalent_diameter']

    test2_1 = np.cbrt(6 * Data_All['Organoids_Volume_Fill'] / np.pi)
    test2_2 = np.cbrt(6 * undersampled_df['Organoids_Volume_Fill'] / np.pi)

    test3_1 = np.cbrt(6 * Data_All['Cavity_Volume'] / np.pi)
    test3_2 = np.cbrt(6 * undersampled_df['Cavity_Volume'] / np.pi)

    save_path = ('E:/student/Private/student11/Measure/Measure_copy/Data/ICC005_20240424/undersampling'
                 '/ICC005_0424_undersampling.xlsx')
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)
    undersampled_df.to_excel(save_path, index=False)
    print('Complete!  {}'.format(path))

    # # 打印结果统计
    # print("原始数据分布：")
    # print(pd.cut(orignin_equi_diameter,
    #              bins=[0,10,20,np.inf]).value_counts())
    
    # print("\n降采样后数据分布: ")
    # print(pd.cut(undersampled_df['equivalent_diameter'],
    #              bins=[0,10,20,np.inf]).value_counts())
    


    # 绘制直方图
    # plt.figure(figsize=(9,7))
    # plt.hist(orignin_equi_diameter, bins=50, alpha=0.5, label='Orginal', color='#2E86C1', rwidth=0.8)
    # plt.xlabel('Equivalent Diameter (pixel)', fontsize=20)
    # plt.ylabel('Count', fontsize=20)
    # # plt.legend()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.tight_layout()
    # plt.savefig('Origin.png')

    # plt.figure(figsize=(9,7))
    # plt.hist(undersampled_df['equivalent_diameter'], bins=50, alpha=0.5, label='Undersampled', color='#E67E22', rwidth=0.8)
    # plt.xlabel('Equivalent Diameter (pixel)', fontsize=20)
    # plt.ylabel('Count', fontsize=20)
    # # plt.legend()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.tight_layout()
    # plt.savefig('Undersample.png')


    # # 绘制体积直方图
    # plt.figure(figsize=(9,7))
    # plt.hist(np.cbrt(6 * Data_All['Organoids_Volume_Fill'] / np.pi), bins=50, alpha=0.5, label='Orginal', color='#2E86C1', rwidth=0.8)
    # plt.xlabel('Equivalent Diameter (pixel)', fontsize=20)
    # plt.ylabel('Count', fontsize=20)
    # # plt.legend()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.tight_layout()
    # plt.savefig('volume_origin.png')

    # plt.figure(figsize=(9,7))
    # plt.hist(np.cbrt(6 * undersampled_df['Organoids_Volume_Fill'] / np.pi), bins=50, alpha=0.5, label='Undersampled',color='#E67E22', rwidth=0.8)
    # plt.xlabel('Equivalent Diameter (pixel)', fontsize=20)
    # plt.ylabel('Count', fontsize=20)
    # # plt.legend()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.tight_layout()
    # plt.savefig('volume_undersampled.png')


    # # 绘制最长轴直方图
    # plt.figure(figsize=(9,7))
    # plt.yscale('log')
    # plt.hist(np.cbrt(6 * Data_All['Cavity_Volume'] / np.pi), bins=50, alpha=0.5, label='Orginal',color='#2E86C1', rwidth=0.8)
    # plt.xlabel('Equivalent Diameter (pixel)', fontsize=20)
    # plt.ylabel('Count/logarithmic display', fontsize=20)
    # # plt.legend()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.tight_layout()
    # plt.savefig('cavity_log_origin.png')

    # plt.figure(figsize=(9,7))
    # plt.yscale('log')
    # plt.hist(np.cbrt(6 * undersampled_df['Cavity_Volume'] / np.pi), bins=50, alpha=0.5, label='Orginal',color='#E67E22', rwidth=0.8)
    # plt.xlabel('Equivalent Diameter (pixel)', fontsize=20)
    # plt.ylabel('Count/logarithmic display', fontsize=20)
    # # plt.legend()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.tight_layout()
    # plt.savefig('cavity_log_undersampled.png')