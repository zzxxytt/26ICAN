# 单个目标检测结果。
# 坐标系约定：xmin/xmax 沿图像宽度方向，ymin/ymax 沿图像高度方向。
# 原点 (0,0) 在图像左上角，单位是像素（与 OpenCV 一致）。

# 类别索引，对应 class_names_file 参数所指文件里 names 列表的下标
# （默认是 recognize_yolo/config/classes_coco.txt）
int32 class_id

# 类别名称，例如 "person"、"bottle"
string class_name

# 置信度，范围 0.0 ~ 1.0
float32 score

# 检测框
float32 xmin
float32 ymin
float32 xmax
float32 ymax

# 框中心点，方便跟随类控制节点直接使用，避免重复计算
float32 center_x
float32 center_y

# 归一化中心点，范围 0.0 ~ 1.0，用于与分辨率无关的控制律
float32 norm_center_x
float32 norm_center_y

# 预留跟踪 ID；当前无跟踪器时恒为 -1
int32 track_id
