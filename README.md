# [2024] LGDisplay-DGU-cooperation
- 주제: Multi-light 기반 외관검사 알고리즘 개발

- 목적: 결함 분류 (classification)

- 성과1: 결함 분류 정확도 98.5% (테스트 데이터셋: 192장/195장), 정상 분류 정확도 51.3% (테스트 데이터셋: 157장/306장)

- 성과2: 결함 분류 논문 작성 중 (AVAF-Net)

- 블로그 설명: https://johyeongseob.tistory.com/55

산학과제 코드 설명

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


 - 타임라인
<img width="1076" alt="Image" src="https://github.com/user-attachments/assets/1add0d21-93d5-4fb0-a4f1-f5f009608fef" />
