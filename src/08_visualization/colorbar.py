import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def create_thin_colorbar():
    # 1. 定义 Amira "Physics" 颜色 (蓝-青-绿-黄-红)
    colors = [
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 0.0)
    ]
    cm = mcolors.LinearSegmentedColormap.from_list(name='amira_physics', colors=colors, N=256)

    # 2. 设置绘图区域 (控制长宽比)
    # figsize=(1, 6) 创建一个窄长的画布
    fig = plt.figure(figsize=(1, 6))

    # 手动添加 axes [left, bottom, width, height] (范围 0-1)
    # width=0.15 控制色条本身的宽度 (非常细)
    ax = fig.add_axes([0.1, 0.05, 0.15, 0.9])

    # 3. 设置范围 0-8 (修改处)
    norm = mcolors.Normalize(vmin=0, vmax=6)

    # 4. 绘制 Colorbar
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cm),
        cax=ax,
        orientation='vertical'
    )

    # 5. 设置刻度 (修改处)
    # 生成从 0 到 8 的 5 个刻度点: [0, 2, 4, 6, 8]
    ticks = np.linspace(start=0, stop=6, num=7)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=12)

    # 6. 添加单位 (LaTeX 格式) (修改处)
    # 将单位改为 mm^-1
    cb.set_label(label=r'$mm^{-1}$', fontsize=14, labelpad=5)

    # 7. 保存
    output_filename = 'colorbar_thin_unit_mm.png'
    # bbox_inches='tight' 自动裁剪多余白边
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    print(f"生成的图片已保存为: {output_filename}")

if __name__ == "__main__":
    create_thin_colorbar()