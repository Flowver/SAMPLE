# 아직 UI와 실행파일의 시퀀스 연결이 되어있지 않아요! 젭알 연결시켜주세요ㅠ
# 따라서 실행하기 위해선 1.circle~3.line로 도형별로 정리된 파일에 들어가시면 추가옵션이 각각 적용된 .py 파일을 보실 수 있습니다.

아래는 실행에 앞서 해야할 일들입니다.
(1) 라이브러리 설치하기 (파이썬 버전 : 3.6 (64비트)) -
0. cmd 실행하여 명령어 실행하기
1. pip 업데이트
python -m pip install --upgrade pip
2. numpy 라이브러리 설치
pip install numpy
3. opencv 라이브러리 설치
pip install opencv-python
4. skimage 라이브러리 설치
pip install scikit-image
5. scipy 라이브러리 설치
pip install scipy

(2) 실행하기
0. cmd 실행하여 명령어 실행하기
1. [.py] 파일이 포함된 폴더로 cd 명령어로 이동하기
ex) cd C:\Users\cau\Documents\GitHub\SAMPLE\file\genetic_image-master\1. circle
2. 원하는 코드 실행하기 (only jpg)
python 원하는거.py 불러온이미지이름.jpg
ex) python circle_.py jpg.jpg