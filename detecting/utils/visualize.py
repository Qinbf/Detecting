import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from detecting.utils.colors import get_color

# 显示检测框
def display_instances(image, boxes, class_ids, class_names, confidence_thresh=0.05,
                      scores=None, title="", figsize=(16, 16), ax=None):
    '''
    image: 显示的图片
    boxes: 检测框坐标
    class_ids: 检测框对应类别编号
    class_names: 检测框对应类别名称
    confidence_thresh: 置信度阈值
    scores: 每个框的置信度
    title: 图片标题
    figsize: 图片大小
    '''
    # 标注框数量
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    # plt.figure(dpi=120)
    # 生成一张图像
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize, dpi=120)

    # 图片的高度和宽度
    height, width = image.shape[:2]
    # 把图像的显示范围设置得大一些
    ax.set_ylim(height + 12, -12)
    ax.set_xlim(-12, width + 12)
    # 不显示坐标
    ax.axis('off')
    # 设置图片标题
    ax.set_title(title)
    # 循环这N个框
    for i in range(N):
        # 如果没有检测框信息
        if not np.any(boxes[i]):
            # 跳过
            continue
        # 如果scores不是None
        if scores is not None:
            # 获得检测框置信度
            score = scores[i]
            if score < confidence_thresh:
                # 跳过
                continue
        else:
            score = None
 
        # 根据种类获取对应颜色
        color = get_color(class_ids[i])
        # 获得检测框坐标
        y1, x1, y2, x2 = boxes[i]
        # 转为int类型
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        # 画矩阵
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=1, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # 获得类别编号
        class_id = class_ids[i]
        # 获得类别名称
        label = class_names[class_id]
        # 文字信息标签名+置信度
        caption = "{} {:.3f}".format(label, score) if score else label
        # 设置文字在检测框左上角
        ax.text(x1+2.3, y1-5, caption,
                color='w', size=12, backgroundcolor=color)
    # 显示图片
    plt.imshow(image.astype(np.uint8))
    plt.show()
    
# 可以传入两组boxes分别显示
def draw_boxes(image, boxes=None, refined_boxes=None,
               captions=None, visibilities=None,
               title="", ax=None):
    '''Draw bounding boxes and segmentation masks with differnt
    customizations.
    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    '''
    # boxes数量
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]
    # 生成一张图像
    if not ax:
        _, ax = plt.subplots(1, figsize=(16, 16), dpi=120)


    # 显示区域比原图片更大一些
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')
    # 设置title
    ax.set_title(title)
    # 循环
    for i in range(N):
        # 设置一些不同风格
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            # 灰色
            color = "gray"
            # 虚线
            style = "dotted"
            # 透明度
            alpha = 0.5
        elif visibility == 1:
            # 获得颜色
            color = get_color(i)
            # 虚线
            style = "dotted"
            # 透明度
            alpha = 1
        elif visibility == 2:
            # 获得颜色
            color = get_color(i)
            # 实线
            style = "solid"
            # 透明度
            alpha = 1

        # 画boxes
        if boxes is not None:
            # 获得坐标
            y1, x1, y2, x2 = boxes[i]
            # 转为int类型
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
            # 画矩阵
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # 画refined_boxes
        if refined_boxes is not None and visibility > 0:
            # 获得坐标
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            # 转为int类型
            ry1, rx1, ry2, rx2 = int(ry1), int(rx1), int(ry2), int(rx2)
            # 画矩阵
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # 画一条线连接boxes和refined_boxes的左上方
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # 显示框的描述
        if captions is not None:
            caption = captions[i]
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

    plt.imshow(image.astype(np.uint8))
    plt.show()