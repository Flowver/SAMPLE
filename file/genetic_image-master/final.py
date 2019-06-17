import cv2, random, os, sys # opencv / 랜덤 / 디렉터리 / 시스템 인자 설

import numpy as np # 행렬

from copy import deepcopy # copy 라이브러리에서 deepcopy 함수만 사용

from skimage.measure import compare_mse # skimage.measure 라이브러리에서 compare_mse 함수 (원본이미지와의 차이점 계산)

import multiprocessing as mp # multiprocessing 라이브러리 사용


def DrawCircle():
  filename, ext = os.path.splitext(os.path.basename(filepath))
  print(filepath);
  img = cv2.imread(filepath)
  width, height, channels = img.shape; #bgr 형식으로 읽는다

  # hyperparameters - 초매개변수
  n_initial_genes = 50 # 첫번째 세대의 개수
  n_population = 50 # 한 세대당 유전자 그룹의 개수
  prob_mutation = 0.01 # 돌연변이가 발생할 확률 1%
  prob_add = 0.3 # 유전자 그
  # 룹에 원이 발생활 확률 30%
  prob_remove = 0.2 # 유전자 그룹에 원을 없앨 확률 20%

  min_radius, max_radius = 5, 15 # 원의 크기는 최소 5에서 최대 15까지
  save_every_n_iter = 100 # 이미지 저장 주기는 100세대 마다

  # Gene
  class Gene(): # 유전자(한개의 동그라미)에 대한 클래스 (위치/색깔/크기)
    def __init__(self):

      # 동그라미가 캔버스안에 위치하는 xy좌표 (width-캔버스 가로길이 / height -캔버스 세로길이)
      self.center = np.array([random.randint(0, width), random.randint(0, height)])

      # 동그라미의 반지름(min_radius-원의 최소, max_radius-원의 최대)
      self.radius = random.randint(min_radius, max_radius)

      # 동그라미의 색깔
      self.color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    def mutate(self): # 돌연변이를 생성합니다
      # 평균 15, 표준편차 4인 분포에서 랜덤한 숫자 추출 -> 100으로 나눠줘..
      # 원래 유전자가 가진 특성값에서 플러스 마이너스 15%만큼 변이한다.

      # 돌연변이의 범위 설정 (mutation_size)
      mutation_size = max(1, int(round(random.gauss(15, 4)))) / 100

      # 원래 유전자가 가진 특성값에서 플러스 마이너스 15%만큼 변이한다.
      r = random.uniform(0, 1)
      if r < 0.33: # radius 33% 확률로 반지름을 변경함
        self.radius = np.clip(random.randint(
          int(self.radius * (1 - mutation_size)),
          int(self.radius * (1 + mutation_size))
        ), 1, 100)

      elif r < 0.66: # center 33% 확률로 위치를 변경함
        self.center = np.array([
          np.clip(random.randint(
            int(self.center[0] * (1 - mutation_size)),
            int(self.center[0] * (1 + mutation_size))),
          0, width),
          np.clip(random.randint(
            int(self.center[1] * (1 - mutation_size)),
            int(self.center[1] * (1 + mutation_size))),
          0, height)
        ])
      else: # color - center 33% 확률로 색깔을 변경함
        self.color = np.array([
          np.clip(random.randint(
            int(self.color[0] * (1 - mutation_size)),
            int(self.color[0] * (1 + mutation_size))),
          0, 255),
          np.clip(random.randint(
            int(self.color[1] * (1 - mutation_size)),
            int(self.color[1] * (1 + mutation_size))),
          0, 255),
          np.clip(random.randint(
            int(self.color[2] * (1 - mutation_size)),
            int(self.color[2] * (1 + mutation_size))),
          0, 255)
        ])

  # compute fitness - 환경에 얼마나 작 적응했는지 판단하는 함수
  def compute_fitness(genome):
      # (height~channels) 이미지의 크기 값을 가지고
      # numpy.ones() - 1로 채워진 배열을 만든다 /
    out = np.ones((height, width, channels), dtype=np.uint8) * 255

      # [유전자를 그린다] cv2.circle() - 원을 그린다 / thickness=-1 원을 색상으로 채운다
    for gene in genome:
      cv2.circle(out, center=tuple(gene.center), radius=gene.radius, color=(int(gene.color[0]), int(gene.color[1]), int(gene.color[2])), thickness=-1)
    # mean squared error
    # compare_mse - 두 이미지의 차이를 계산 / img-원본 / out-결과이미지
    # 생성한 이미지가 원본 이미지와 비슷하다 = mse가 낮다 = fitness가 높다
    fitness = 255. / compare_mse(img, out)
    return fitness, out

  # compute population
  def compute_population(g):
    genome = deepcopy(g)
    # mutation - 세대를 한번에 돌연변이로 만드는 함
    if len(genome) < 200: # 유전자의 개수가 200보다 작을
      for gene in genome:
        if random.uniform(0, 1) < prob_mutation:
          gene.mutate()
    else: # 유전자의 개수가 많아지면 - random.sample(a,k) - a에서 k개만큼 랜덤으로 추출한다.
          # mutation 확률만큼
      for gene in random.sample(genome, k=int(len(genome) * prob_mutation)): #
        gene.mutate()

    # add gene - 유전자 추가
    if random.uniform(0, 1) < prob_add:
      genome.append(Gene())

    # remove gene - 유전자 삭제
    if len(genome) > 0 and random.uniform(0, 1) < prob_remove:
      genome.remove(random.choice(genome))

    # compute fitness - 새로운 유전자의 fitness점수를 측정해 리턴합니다
    new_fitness, new_out = compute_fitness(genome)

    return new_fitness, genome, new_out



### 메인함수
  if __name__ == '__main__':
      # main
      os.makedirs('result', exist_ok=True)

      # cpu개수보다 한 개적은 멀티프로세싱 풀을 만든다(구글링)
      p = mp.Pool(mp.cpu_count() - 1)

      # 1st gene
      # (첫번째 세대의 개수)만큼 반복해 유전자를 생성해 best_genome함수에 넣습니다.
      best_genome = [Gene() for _ in range(n_initial_genes)]

      # 첫번째 유전자가 얼만큼 좋은지 평가합니다
      best_fitness, best_out = compute_fitness(best_genome)

      n_gen = 0

      # 멀티프로세싱을 위해 50개의 유전자를 (한 세대당 유전자 그룹의 개수)만큼 다시 생성해서 compute_population으로 넘겨줌

      while True:
          try:
              results = p.map(compute_population, [deepcopy(best_genome)] * n_population)
          except KeyboardInterrupt:
              p.close()
              break

          results.append([best_fitness, best_genome, best_out])

          # 병렬처리 로직
          new_fitnesses, new_genomes, new_outs = zip(*results)

          # fitness 점수에따라 내림차순으로 정리한다.
          best_result = sorted(zip(new_fitnesses, new_genomes, new_outs), key=lambda x: x[0], reverse=True)

          best_fitness, best_genome, best_out = best_result[0]

          # end of generation
          print('Generation #%s, Fitness %s' % (n_gen, best_fitness))
          n_gen += 1

          # visualize
          # cv2.imwrtie() 이미지를 저장한다
          if n_gen % save_every_n_iter == 0:
              cv2.imwrite('result/%s_%s.jpg' % (filename, n_gen), best_out)

          cv2.imshow('best out', best_out)  # (title,image) 윈도우 창의 title, imread의 return값
          if cv2.waitKey(1) == ord('q'):
              p.close()
              break


from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import *

window=Tk()
window.title("COPYPAINTER by Pinky")
window.geometry("500x360")
window.resizable(width=FALSE, height=FALSE)

filepath = ""

def fileopen():
    temppath=askopenfilename(parent=window, filetypes=(("JPEG(*.jpeg, *.jpg, *.jpe)","*.jpeg *.jpg *.jpe"),
                                                     ("JPEG2000(*.jp2)","*.jp2"),("BMP(*.bmp)","*.bmp"),("DIB(*.dib)","*.dib"),
                                                      ("PNG(*.png)","*.png"), ("Portable image format(*.pbm,*.pgm,*.ppm)","*.pbm *.pgm *.ppm"),
                                                      ("Sun rasters(*.sr,*.ras)","*.sr *.ras"),("TIFF(*.tiff,*.tif)","*.tiff *.tif"),
                                                      ("all files","*.*")))
    return temppath

class Widjets:
      
    ## 메세지 박스 기능 ##
    def messagebox1():
        messagebox.showinfo("제작자☺","몇 가지 설정을 통해 사진을\n그림으로 만드는 프로그램\n\n 제작자 : Pinky\n 문의 : sis04263@naver.com")

    def messagebox2():
        messagebox.showinfo("가이드북☺ ", "[도구 선택]\n선택하신 도구로 그림이 그려집니다.\n\n [추가옵션] \n->테두리만 : 선택한 도구의 내부가 빈 상태 \n->크기제한 : 실행창의 화면 크기가 고정됨.(사진 비율유지) \n->흑백(알파값)모드:알파값은 테두리만 옵션 선택시 사용 가능합니다. 알파값은 선택한 도구의 투명도를 결정하는 값입니다")
        


    ## 라디오 버튼 기능 ##
    def myradio():
        if var.get()==1: #원
            if var2.get()==0: #흑백(0은 컬러 1은 흑백)
                if var3.get()==1: #크기제한
                    if var4.get()==1: #테두리
                        label4.configure(text="테두리 / 크기 제한 / 원 [선택된 도구]")
                    else:
                        label4.configure(text="크기 제한 / 원 [선택된 도구]") # [c2]
                else:
                    if var4.get()==1:
                        label4.configure(text="테두리 / 원 [선택된 도구]") # [c3]
                    else:
                        label4.configure(text="원 [선택된 도구]") # [c4]
            else:
                if var3.get()==1:
                    if var4.get()==1:
                        label4.configure(text="흑백(알파값) / 테두리 / 크기 제한 / 원 [선택된 도구]") # [c5]
                        
                    else:
                        label4.configure(text="흑백(알파값) / 크기 제한 / 원 [선택된 도구]") # [c6]
                else:
                    if var4.get()==1:
                        label4.configure(text="흑백(알파값) / 테두리 / 원 [선택된 도구]") # [c7]
                    else:
                        label4.configure(text="흑백(알파값) / 원 [선택된 도구]") # [c8]
                                        
        elif var.get()==2: #사각형
            if var2.get()==0:#컬
                if var3.get()==1: #크기제한
                    if var4.get()==1: #테두리
                        label4.configure(text="테두리 / 크기 제한 / 사각형 [선택된 도구]") # [rt]
                    else:
                        label4.configure(text="크기 제한 / 사각형 [선택된 도구]") # [rt2]
                else:
                    if var4.get()==1:
                        label4.configure(text="테두리 / 사각형 [선택된 도구]") # [rt3]
                    else:
                        label4.configure(text="사각형 [선택된 도구]") # [rt4]
            else:
                if var3.get()==1:
                    if var4.get()==1:
                        label4.configure(text="흑백(알파값) / 테두리 / 크기 제한 / 사각형 [선택된 도구]") # [rt5]
                    else:
                        label4.configure(text="흑백(알파값) / 크기 제한 / 사각형 [선택된 도구]") # [rt6]
                else:
                    if var4.get()==1:
                        label4.configure(text="흑백(알파값) / 테두리 / 사각형 [선택된 도구]") # [rt7]
                    else:
                        label4.configure(text="흑백(알파값) / 사각형 [선택된 도구]") # [rt8]
                        
        elif var.get()==3: # 선
            if var3.get()==1:#크기제한
                label4.configure(text="크기 제한 / 선 [선택된 도구]") #[line1]
            else:
                label4.configure(text="선 [선택된 도구]") #[line2]

        else: # 선택X
            if var2.get()==0: #컬러
                if var3.get()==1: #크기제한
                    if var4.get()==1: #테두리
                        label4.configure(text="테두리 / 크기 제한 [선택된 도구]")
                    else:
                        label4.configure(text="크기 제한 [선택된 도구]")
                else:
                    if var4.get()==1:
                        label4.configure(text="테두리 [선택된 도구]")
                    else:
                        label4.configure(text="[선택된 도구]")
            else:
                if var3.get()==1:
                    if var4.get()==1:
                        label4.configure(text="흑백(알파값) / 테두리 / 크기 제한 [선택된 도구]")
                        
                    else:
                        label4.configure(text="흑백(알파값) / 크기 제한 [선택된 도구]")
                else:
                    if var4.get()==1:
                        label4.configure(text="흑백(알파값) / 테두리 [선택된 도구]")
                    else:
                        label4.configure(text="흑백(알파값) [선택된 도구]")
                
    ## 시작 버튼 기능 ##
    def followtheradio():
        if var.get()==1: #원
            if var2.get()==0: #컬러
                if var3.get()==1: #크기제한
                    if var4.get()==1: #테두리
                        c_color_rs_l()
                    else:
                        c_color_rs() # [c2]
                else:
                    if var4.get()==1:
                        c_coior_1() # [c3]
                    else:
                        c_color() # [c4]
            else:
                if var3.get()==1:
                    if var4.get()==1:
                        c_rs_1() # [c5]
                        
                    else:
                        c_rs() # [c6]
                else:
                    if var4.get()==1:
                        c_1() # [c7]
                    else:
                        c() # [c8]                                        
        elif var.get()==2: #사각형
            if var2.get()==0:#컬
                if var3.get()==1: #크기제한
                    if var4.get()==1: #테두리
                        rt_color_rs_l() # [rt]
                    else:
                        rt_color_rs() # [rt2]
                else:
                    if var4.get()==1:
                        rt_coior_1() # [rt3]
                    else:
                        rt_color() # [rt4]
            else:
                if var3.get()==1:
                    if var4.get()==1:
                        rt_rs_1() # [rt5]
                    else:
                        rt_rs() # [rt6]
                else:
                    if var4.get()==1:
                        rt_1() # [rt7]
                    else:
                        rt() # [rt8]
                        
        elif var.get()==3: #선
            if var3.get()==1:#크기제한
                line_l() #[line1]
            else:
                line() #[line2]

        else:
            messagebox.showerror("경고","도구를 선택해주세요!")

    ## 붓도구 커맨드 ##                        
def c_color_rs_l(): global filepath ; filepath=fileopen(); print(filepath); DrawCircle();
def c_color_rs():global filepath ; filepath=fileopen(); DrawCircle();
def c_coior_1():global filepath ; filepath=fileopen(); DrawCircle();
def c_color():global filepath ; filepath=fileopen(); DrawCircle();
def c_rs_1():global filepath ; filepath=fileopen(); DrawCircle();
def c_rs():global filepath ; filepath=fileopen(); DrawCircle();
def c_1():global filepath ; filepath=fileopen(); DrawCircle();
def c():global filepath ; filepath=fileopen(); DrawCircle();
def rt_color_rs_l():fileopen(); global filepath ; filepath=fileopen(); circle.filepath();
def rt_color_rs():fileopen();global filepath ; filepath=fileopen(); circle.filepath();
def rt_coior_1():fileopen(); None
def rt_color():fileopen(); None
def rt_rs_1():fileopen(); None
def rt_rs():fileopen(); None    
def rt_1():fileopen(); None
def rt():fileopen(); None
def line_l():fileopen(); None
def line():fileopen(); None            

'''----------------------------------------------------------------------------------------------------------------------------------'''

## [tkinter 디자인 부분] ##
label8 = Label(window, bg="#a2b4c6", width=500, height=360)

## 이미지 부분 ##
photo=PhotoImage()

pLabel=Label(window,image=photo)

background_3=PhotoImage(file='file2.gif')

pLabel.configure(image=background_3)

pLabel.place(x=0,y=0)

## 그림도구 선택 ##

label7 = Label(window, font=("나눔스퀘어 ExtraBold", 13), bg="#A2B4C6", width=22, height=16)
label4=Label(window, text="[현재 선택된 것들]",font=("나눔스퀘어 ExtraBold", 13), bg="white", width=25)
label5=Label(window, text="[추가 옵션]",font=("나눔스퀘어 ExtraBold", 13), bg="#7594b9", width=20)
label6=Label(window, text="[도구 선택]",font=("나눔스퀘어 ExtraBold", 13), bg="#7594b9", width=20)



## 시작/종료 ##
button1=Button(window, text="실행", font=("나눔스퀘어 ExtraBold", 13),command=Widjets.followtheradio)
button2=Button(window, text="종료", font=("나눔스퀘어 ExtraBold", 13),command=quit)

## 메인 코드 부분
mainMenu=Menu(window)
window.config(menu=mainMenu)

fileMenu=Menu(mainMenu)
mainMenu.add_command(label="프로그램 정보(INFO)",command=Widjets.messagebox1)
mainMenu.add_command(label="사용설명서(GUIDE)", command=Widjets.messagebox2)

 ## 라디오 버튼
var=IntVar()
rb1=Radiobutton(window, text="원 ●", font=("나눔스퀘어 ExtraBold", 13), variable=var, value=1, command=Widjets.myradio,bg="#a2b4c6")
rb2=Radiobutton(window, text="사각형 ■", font=("나눔스퀘어 ExtraBold", 13), variable=var, value=2, command=Widjets.myradio,bg="#a2b4c6")
rb3=Radiobutton(window, text="선 〓", font=("나눔스퀘어 ExtraBold", 13), variable=var, value=3, command=Widjets.myradio,bg="#a2b4c6")

 ## 체크 버튼
var2=IntVar()
cb1=Checkbutton(window, text="흑백(알파값)모드", font=("나눔스퀘어 ExtraBold", 13), variable=var2,
                selectcolor='#B7F0B1', bg='white', relief='raised',overrelief='sunken',command=Widjets.myradio)

var3=IntVar()
cb2=Checkbutton(window, text="크기 제한 (가로 400px)", font=("나눔스퀘어 ExtraBold", 13), variable=var3,
                selectcolor='#FFA7A7', bg='white', relief='raised',overrelief='ridge',command=Widjets.myradio)


var4=IntVar()
cb3=Checkbutton(window, text="테두리만", font=("나눔스퀘어 ExtraBold", 13), variable=var4,
                selectcolor='#B2CCFF', bg='white', relief='raised',overrelief='sunken',command=Widjets.myradio)

## pack이 아니라 place함수 사용해서 절대위치로 다시 수정할 것 / place 사용이 나중에 수정이 용이함

label8.place(x=0,y=0)
pLabel.place(x=31,y=23)

label4.place(x=0, y=338, width=500)
label5.place(x=42, y=135)
label6.place(x=42, y=30)



cb3.place(x=40, y=165, width=210)
cb2.place(x=40, y=200, width=210)
cb1.place(x=40, y=235, width=210)

rb3.place(x=115, y=55)
rb2.place(x=115, y=80)
rb1.place(x=115, y=105)

button1.place(x=40, y=273, width=100)
button2.place(x=150, y=273, width=100)



window.mainloop()

