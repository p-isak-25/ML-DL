# ML_virtual_ku

파이썬 데이터 처리, PyTorch 회귀·분류, Gymnasium 강화학습 실습을 모은 프로젝트입니다. 분류 모델과 MNIST 실습은 CPU에서 학습합니다.

## 시작하기

프로젝트 루트에서 기존 가상환경을 활성화합니다. 현재 가상환경은 Python 3.10으로 구성되어 있으며, VS Code 기본 인터프리터는 `.venv/bin/python`입니다.

```bash
source .venv/bin/activate
python -m pip check
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.ones(1).device)'
```

마지막 명령은 `2.7.1+cpu 0.22.1+cpu`와 `cpu`를 출력합니다. VS Code에서 다른 인터프리터를 이미 선택했다면 **Python: Select Interpreter**에서 프로젝트의 `.venv/bin/python`을 선택합니다.

### 가상환경 다시 만들기

아래 명령은 Linux에서 새로 설정할 때 사용합니다. GridWorld 창을 열려면 시스템 Python에 Tkinter가 필요합니다. Ubuntu에서 Tkinter가 빠져 있다면 `python3-tk` 시스템 패키지를 설치합니다.

```bash
python3.10 -m venv --prompt ML_virtual_ku .venv
source .venv/bin/activate
python -m pip install numpy==2.2.6 pandas==2.3.3 matplotlib==3.10.9 \
  opencv-python-headless==5.1.0.95 openpyxl==3.1.5 \
  'gymnasium[classic-control,toy-text]==1.1.1' pygame==2.6.1 \
  pillow==12.3.0 scikit-learn==1.7.2
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu'
python -m pip install stable-baselines3==2.6.0 ale-py==0.11.2 tensorboard==2.21.0
```

## 주차별 실습

각 파일을 프로젝트 루트에서 `python practice/week_XX/<파일명>` 형태로 실행합니다.

| 파일 | 실습 내용 |
| --- | --- |
| [2week.py](practice/week_02/2week.py) | NumPy 배열과 슬라이싱, pandas 표와 엑셀, OpenCV 이미지 읽기, Matplotlib 그래프 |
| [3week_SLR.py](practice/week_03/3week_SLR.py) | 입력 특성 하나를 사용하는 단순 선형 회귀 |
| [3week_MLR.py](practice/week_03/3week_MLR.py) | 여러 입력 특성을 사용하는 다중 선형 회귀 |
| [4week_LogistinR.py](practice/week_04/4week_LogistinR.py) | 시그모이드와 이진 교차 엔트로피를 사용하는 로지스틱 회귀 |
| [4week_softmax.py](practice/week_04/4week_softmax.py) | 소프트맥스 확률, 원-핫 인코딩, 다중 클래스 손실 계산 |
| [4week_softmaxR_test.py](practice/week_04/4week_softmaxR_test.py) | 직접 계산한 손실과 PyTorch 교차 엔트로피로 다중 클래스 분류 모델 학습 |
| [4week_mnist.py](practice/week_04/4week_mnist.py) | 손글씨 숫자 분류 모델을 15에폭 학습하고 테스트 세트 정확도와 예측 표시 |
| [9week_gymnasium.py](practice/week_09/9week_gymnasium.py) | CartPole-v1에서 무작위 행동을 선택하며 환경 실행 |
| [9week_incremental.py](practice/week_09/9week_incremental.py) | 같은 보상 시퀀스에서 전체 평균 계산과 증분 평균 계산 비교 |
| [10week_policy_iteration.py](practice/week_10/10week_policy_iteration.py) | GridWorld에서 정책 평가와 정책 개선을 번갈아 수행 |
| [10week_value_iteration.py](practice/week_10/10week_value_iteration.py) | GridWorld에서 상태 가치를 갱신하고 가장 높은 가치의 행동 선택 |

`2week.py`는 `dog.jpg`, `Example.xlsx`, `spam.xlsx`를 스크립트와 같은 [week_02](practice/week_02) 폴더에서 읽습니다. 파일이 없으면 예제 데이터로 대체합니다. `4week_mnist.py`는 [practice/week_04/MNIST_data](practice/week_04/MNIST_data)의 데이터를 사용합니다.

### Google Drive에서 가져온 실습 자료

Drive의 실습 자료는 [practice](practice)의 `week_XX/`에 있습니다. 9주차와 10주차의 실습 코드도 해당 주차 폴더에 있습니다. 12~14주차는 [practice/common](practice/common)의 GridWorld 환경과 시각화 모듈을 공유합니다.

| 경로 | 자료 |
| --- | --- |
| [week_02](practice/week_02) | 2주차 PDF, 이미지, 엑셀, NumPy 메모 |
| [week_03](practice/week_03) ~ [week_07](practice/week_07) | 3~7주차 실습 PDF |
| [week_09](practice/week_09) | 발표 자료와 [Bandit 실습](practice/week_09/bandit.py) |
| [week_10](practice/week_10) | 정책 반복과 가치 반복 발표 자료. 실행 코드는 위 표, 이미지는 `gridworld/img/`에 있습니다. |
| [week_12](practice/week_12) | Monte Carlo와 TD 실습 |
| [week_13](practice/week_13) | Q-learning과 SARSA 실습 |
| [week_14](practice/week_14) | 신경망 Q-learning과 DQN 실습 |
| [week_15](practice/week_15) | Atari DQN과 Stable-Baselines3 예제 |
| [reference](practice/reference) | Transformer 한국어 참고 PDF |

### 추가 강화학습 예제 실행

프로젝트 루트에서 아래 명령을 실행합니다. Matplotlib 그래프 없이 실행할 때는 명령 앞에 `MPLBACKEND=Agg`를 붙입니다. 14주차 DQN과 15주차 예제에서 `--headless`를 빼면 평가 과정을 창에서 볼 수 있습니다.

```bash
python practice/week_09/bandit.py
python practice/week_12/MC.py
python practice/week_12/TD.py
python practice/week_12/MCAgent.py
python practice/week_12/TDAgent.py
python practice/week_13/Q-learning.py
python practice/week_13/Q-learning_simple.py
python practice/week_13/SARSA.py
python practice/week_14/Q-learning_nn.py
python practice/week_14/DQN.py --headless
python practice/week_15/bl3_ex.py --headless
python practice/week_15/bl_dqn_atari.py --headless
```

Atari DQN은 CPU에서 10,000스텝을 학습하며 실행한 디렉터리에 `dqn_atari.zip`과 `logs/dqn_cnn/`을 생성합니다. 빠르게 학습과 평가를 확인할 때는 `--timesteps 300 --eval-episodes 1 --eval-steps 50`을 추가합니다.

## GridWorld

두 실습은 동일한 5×5 격자에서 상하좌우 이동을 비교합니다. 격자 좌표 `[2, 2]`는 보상 +1인 종료 상태이며, `[1, 2]`와 `[2, 1]`은 진입 시 보상 −1을 주는 칸입니다.

- **정책 반복**은 [environment_policy.py](gridworld/environment_policy.py)의 `Env`와 `GraphicDisplay`를 사용합니다. 현재 정책을 기준으로 상태 가치를 평가한 뒤, 계산한 가치로 정책을 개선합니다.
- **가치 반복**은 [environment_value.py](gridworld/environment_value.py)의 `Env`와 `GraphicDisplay`를 사용합니다. 가능한 행동 중 최댓값으로 상태 가치를 갱신하고, `Print Policy`로 현재 가치에 따른 행동을 표시합니다.
- [gridworld/img](gridworld/img)에는 두 Tkinter 창에서 사용하는 캐릭터, 보상 칸, 방향 화살표 이미지가 있습니다.

```bash
python practice/week_10/10week_policy_iteration.py
python practice/week_10/10week_value_iteration.py
```

## Taxi 과제

모두 Gymnasium의 Taxi-v3를 사용합니다. Taxi-v3는 상태 500개와 행동 6개를 제공하며, 정책 반복 구현은 환경의 전이 테이블에서 보상과 다음 상태를 읽습니다.

| 파일 | 역할 |
| --- | --- |
| [policyiteration.py](taxi_assignment/policyiteration.py) | Q2와 Q3가 공유하는 정책 평가·개선 클래스 |
| [Q2.py](taxi_assignment/Q2.py) | 정책 개선 전에 100번 평가하고, 안정화 후 전체 상태의 정책과 가치 출력 |
| [Q3.py](taxi_assignment/Q3.py) | 개선당 평가 10회와 1회를 비교하고 상태별 가치와 행동의 일치 개수 출력 |
| [Q4.py](taxi_assignment/Q4.py) | 가치 변화량을 기준으로 평가를 종료하는 자체 정책 반복 구현 |
| [HW2_gymnasium_taxi.py](taxi_assignment/HW2_gymnasium_taxi.py) | 무작위 행동으로 Taxi 환경을 시각화 |

```bash
python taxi_assignment/Q2.py
python taxi_assignment/Q3.py
python taxi_assignment/Q4.py
python taxi_assignment/HW2_gymnasium_taxi.py
```

## 이진 분류 프로젝트

[classification_project](classification_project)의 데이터는 모두 `x,y,label` 열을 사용합니다.

| 파일 | 역할 |
| --- | --- |
| [데이터 생성 코드](classification_project/project_data_code_202112333_박종윤.py) | 좌표를 생성하고 비선형 조건으로 이진 라벨을 지정한 뒤 CSV와 산점도를 저장 |
| [생성된 CSV](classification_project/project_data_202112333_박종윤.csv), [산점도](classification_project/project_data_image_202112333_박종윤.png) | 데이터 생성 코드의 출력물 |
| [dataA.csv](classification_project/dataA.csv), [dataB.csv](classification_project/dataB.csv), [dataC.csv](classification_project/dataC.csv) | 모델 학습에 사용할 수 있는 입력 데이터. 현재 입력은 `dataC.csv`입니다. |
| [모델 학습 코드](classification_project/project_model_code_202112333_박종윤_.py) | 좌표를 표준화하고 CPU에서 이진 분류기를 100에폭 학습하며 손실 출력 |

```bash
python classification_project/project_model_code_202112333_박종윤_.py
```

데이터를 새로 생성하려면 `python classification_project/project_data_code_202112333_박종윤.py`를 실행합니다. 이 명령은 같은 폴더의 생성 CSV와 PNG를 덮어씁니다. 모델 학습은 기존 `dataC.csv`로 바로 실행할 수 있습니다.

## 실행 확인 범위

- CPU 전용 PyTorch에서 MNIST 15에폭과 분류 모델 100에폭을 완료했습니다.
- GridWorld는 실제 Tkinter 창에서 가치 계산과 정책 평가·개선을 확인했습니다.
- CartPole과 Taxi 시각화 예제는 가상 화면에서 실행을 확인했습니다. 실제 창을 보려면 그래픽 세션에서 실행합니다.
- 2주차 실습은 Drive에서 가져온 원본 이미지와 엑셀 데이터로 확인했습니다.
