# 方案
  ## 概述
    1. 读入视频文件，对每一帧进行目标检测，检测person/shoot/made；
    2. 根据made事件，找到对应的shoot事件，根据shoot找到对应的person；
    3. 然后将shoot到made之间视频加上一定缓冲，按照指定格式输出；
  ## 模型
    1. 目标检测模型使用yolo8x
       1. 使用定制篮球数据集训练，支持检测篮球/人/shoot/made 等几种类型
          1. 参见 https://universe.roboflow.com/ownprojects/basketball-w2xcw/dataset/2
          2. 训练脚本参见 [yolo8-train](../python/train/yolo8-train.ipynb)
    2. 集成boxmot，支持多种检测模型和reid模型（不过最终只使用了其中的reid模型）
    3. reid模型使用boxmot内置的clip_market1501.pt，用于person匹配
       1. 也集成了clipreid，不过缺省模型的size不匹配，需要重新训练才能使用

# 测试
    1. 视频文件： 3.77G , 时间13:33，48750 frames，made point=40
    2. 在kaggle/t4 上运行结果：
       1. 耗时=16200s, find result len=46（40 made都找到）
       2. Precision： 40/46*100=86.9565
       3. Recall ： 40/40*100=100
# 总结
    1. 基本上能够满足需求，但是还有很多可以优化的地方
       1. 性能：
          1. 耗时太长，需要优化，目前看主要是yolov8x目标检测的耗时太长
          2. yolov8n可以快上4-5倍，但是精度差一下，有些没有检测出来
       2. 精度：
          1. 有些made判断不准确，需要优化，比如有些made实际都是没有进球，仅仅是球从篮筐前后落下而已
          2. 有些shoot的person匹配不准确，需要优化，比如有些shooter 被防守人挡住了，目前算法查找的时候实际上是找到了防守人
