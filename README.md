# BasketballAI

## overview
    个人非常喜欢打篮球，球友中有喜欢录像的，每次在录像中找自己的进球集锦比较麻烦，所以产生了利用ai视觉技术自动生成进球集锦的想法；
    项目的初级目标是，通过ai技术，自动将录像中的进球集锦剪辑出来，并按照人分类/过滤来生成视频；
    考虑到个人目前休业在家，也不排除将来将这个项目做成一个产品的可能性；
    项目将按阶段进行，每个阶段都有一个明确的目标，每个阶段都有一个demo，每个阶段都有一个明确的时间点和结果，详细参见 design/plan.md
    目前初步完成stage1， 详细参见 [design/stage1.md](./design/stage1.md)
## stage1
### 目标
    1. highlight功能，主要是将视频拆分成进球集锦，一个进球一个视频，包括进球的人，进球的时间；
    2. 能够按照人生成视频，比如输入一些player照片信息来过滤视频，比如输入`player1,player2`，则生成的视频中只有`player1,player2`的进球；
    3. 第一版简单起见，有如下假设
       1. 1个固定摄像头录像，多个文件按照时间顺序拼接；
       2. 比赛模式，即只有一个球在场上，每次也只有一个投篮和进球；
       3. 视频文件仅支持本地文件，不支持网络文件；
    4. 时间：9-10月

### 使用说明
    1. 参见[python/demo_usage.ipynb](./python/demo_usage.ipynb)
    2. 主要步骤：
       1. [python/extract_body.py](./python/extract_body.py)提取照片中的人体信息，保存到output/body目录中，然后按照指定格式手动拷贝到input/body下，如果没有人体信息，则生成所有人的进球集锦
          1. 文件名格式：input/body/[player_name]_seq.png
       2. [python/extract_highlights.p](./python/extract_highlight.py)提取视频中的进球集锦，保存到output/highlights目录中
          1. 文件名格式： output/highlights/[player_name]_[made_seq]_[made_time].mp4
