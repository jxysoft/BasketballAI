## stage1
### 目标
    1. highlight功能，主要是将视频拆分成进球集锦，一个进球一个视频，包括进球的人，进球的时间；
    2. 能够按照人生成视频，比如输入一些player照片信息来过滤视频，比如输入`player1,player2`，则生成的视频中只有`player1,player2`的进球；
    3. 第一版简单起见，有如下假设
       1. 1个固定摄像头录像，多个文件按照时间顺序拼接；
       2. 比赛模式，即只有一个球在场上，每次也只有一个投篮和进球；
       3. 视频文件仅支持本地文件，不支持网络文件；
    4. 时间：9-10月
### 状态
    1. 已完成

## stage2
### 目标
    1. 第一阶段遗留问题解决；
    2. 视频文件支持网络文件（主要是百度网盘）
    3. 支持人脸识别
### 状态
    1. 未开始，11月主要投入参加一些线上比赛，预计12月底完成

## stage3
### 目标
    1. 统计得分（1，2，3），统计命中旅和位置；
    2. 能在手机android端安装执行，或者可以在huggingface提供服务；
### 状态
    1. 未开始

## stage4
### 目标
    1. 解决准确性/性能等关键问题
    2. todo：考虑是否产品化
    准确