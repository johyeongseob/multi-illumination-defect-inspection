# [2024] LGDisplay-DGU-cooperation
Multi-light 기반 외관검사 알고리즘 개발

- Ensemble 디렉토리
    - model1_ensemble.py: 모델1에 대한 앙상블 결과를 확인할 수 있는 코드입니다.
    - model2_ensemble.py: 모델2에 대한 앙상블 결과를 확인할 수 있는 코드입니다.
    - model3_ensemble.py: 모델3에 대한 앙상블 결과를 확인할 수 있는 코드입니다.
    - model_ensemble.py: 최종 앙상블 결과를 확인할 수 있는 코드입니다.
    - model_train.py: 모델별 학습이 가능한 코드입니다.
    - model_evalutaion.py: 모델별 검증이 가능한 코드입니다.

 - models 디렉토리
    - PretrainedSqueezeNet.py: ImageNet-1k로 사전 훈련된 SqueezeNet 모델입니다.
    - SENet.py: SENet 모델입니다.

 - Classifier.py: 분류작업을 수행하는 코드입니다.
  - model 별 classifier 구조가 변경되도록 파라미터(num_classes)를 사용하였습니다.

 - DataLoader.py: 이미지를 배치 단위로 구성하는 코드입니다.
  - model 별 dataloader 구조가 변경되도록 파라미터(target)를 사용하였습니다.
