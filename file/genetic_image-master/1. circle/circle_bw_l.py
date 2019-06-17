# 원 / 흑백 / 테두리 / 

import cv2, random, os # opencv / 랜덤 / 디렉터리
import numpy as np # 행렬 
from copy import deepcopy # copy 라이브러리에서 deepcopy 함수만 사용
from skimage.measure import compare_mse # skimage.measure 라이브러리에서 compare_mse 함수 (원본이미지와의 차이점 계산)
import multiprocessing as mp # multiprocessing 라이브러리 사용

filepath =('C://Users//sis04//Desktop//file//genetic_image-master//1. circle//jpg.jpg')
filename, ext = os.path.splitext(os.path.basename(filepath))
img = cv2.imread(filepath)
edges = cv2.Canny(img, threshold1=100, threshold2=150)
edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations = 1)
height, width, channels = img.shape #bgr 형식으로 읽는다
    
# hyperparameters - 초매개변수
n_initial_genes = 50 # 첫번째 세대의 개수
n_population = 50 # 한 세대당 유전자 그룹의 개수
prob_mutation = 0.01 # 돌연변이가 발생할 확률 1%
prob_add = 0.3 # 유전자 그룹에 원이 발생활 확률 30%
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

# compute fitness - 환경에 얼마나 작 적응했는지 판단하는 함수
def compute_fitness(genome):
    # (height~channels) 이미지의 크기 값을 가지고
    # numpy.ones() - 1로 채워진 배열을 만든다 / 
  out = np.zeros((height, width), dtype=np.uint8) * 255

    # [유전자를 그린다] cv2.circle() - 원을 그린다 / thickness=-1 원을 색상으로 채운다
  for gene in genome:
    cv2.circle(out, center=tuple(gene.center), radius=gene.radius, color=(255,255,255), thickness=1,lineType=cv2.LINE_AA)

  # mean squared error
  # compare_mse - 두 이미지의 차이를 계산 / img-원본 / out-결과이미지
  # 생성한 이미지가 원본 이미지와 비슷하다 = mse가 낮다 = fitness가 높다
  fitness = 255. / compare_mse(edges, out)

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

# main
if __name__ == '__main__':
  os.makedirs('result', exist_ok=True)

  # cpu개수보다 한 개적은 멀티프로세싱 풀을 만든다(구글링ㄱ)
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

    cv2.imshow('best out', best_out) # (title,image) 윈도우 창의 title, imread의 return값 
    if cv2.waitKey(1) == ord('q'):
     p.close()
     break

  cv2.imshow('best out', best_out)
  cv2.waitKey(0)

