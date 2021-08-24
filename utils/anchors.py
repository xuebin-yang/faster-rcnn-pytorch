import numpy as np

# 以中心点为原点，得到 9 个先验框
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])   # 对于一种scale(尺度)来说的anchor，面积是一样的
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.   # 先验框的左上角 x 坐标
            anchor_base[index, 1] = - w / 2.    # 先验框的左上角 y 坐标
            anchor_base[index, 2] = h / 2.  # 先验框的右下角 x 坐标
            anchor_base[index, 3] = w / 2.  # 先验框的右下角 y 坐标
    return anchor_base

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):  # 这一步做完可以在50*50的feature_map的每一个点上都有9个以该点为中心的anchor_base
    # 计算网格中心点（在筛选proposal anchor之前，必须把输出数据中的中心坐标和长宽转化为真正的坐标和长宽）
    shift_x = np.arange(0, width * feat_stride, feat_stride)  # [1, 50]
    shift_y = np.arange(0, height * feat_stride, feat_stride) # [1, 50]
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)   # 这里是构建中心点坐标系  [50, 50]
    shift = np.stack((shift_x.ravel(),shift_y.ravel(),  # ravel就是展平，和flatten功能一样，但是会影响 shift_x 原本的值
                      shift_x.ravel(),shift_y.ravel(),), axis=1)   # [2500, 4]

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((K, 1, 4))   # [2500, 9, 4]，相当于shift的每一行都广播成9*4去和anchor base相加
    # 所有的先验框，(anchor相对于原图来说的，原图每隔16个像素点就有以该像素点为中心的9个anchor)
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)   # [2500*9, 4]
    return anchor
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_anchor_base()
    print(nine_anchors)

    height, width, feat_stride = 38,38,16
    anchors_all = _enumerate_shifted_anchor(nine_anchors,feat_stride,height,width)
    print(np.shape(anchors_all))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]
    
    for i in [108,109,110,111,112,113,114,115,116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    
    plt.show()
