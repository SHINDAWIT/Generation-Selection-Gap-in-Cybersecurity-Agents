# 기존 모델·엔진·벤치마크와 인용 연구

이 연구는 아래 기존 공개 자원을 활용했습니다. 모델·추론 엔진·벤치마크 자체의 개발과 이 저장소의 평가 코드·분석을 구분합니다.

| 자원 | 연구에서 사용한 범위 | 출처 |
| --- | --- | --- |
| IBM Granite 4.2 8B GGUF | Q4_K_M 모델 | [모델 카드](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF) |
| llama.cpp | 로컬 추론, b11026 | [공식 저장소](https://github.com/ggml-org/llama.cpp), [고정 릴리스](https://github.com/ggml-org/llama.cpp/releases/tag/b11026) |
| CyBench | 정적 파일로 답할 수 있는 개발 15·미사용 11 하위 질문 | [공식 저장소](https://github.com/andyzorigin/cybench), commit `1097a7226eb034d3821208114da38f10b8627ab1` |
| NYU CTF Bench | 외부 정적 최종 플래그 7과제 | [공식 저장소](https://github.com/NYU-LLM-CTF/NYU_CTF_Bench) |
| InterCode-CTF | 외부 정적 최종 플래그 28과제 | [공식 저장소](https://github.com/princeton-nlp/intercode) |

가중치·실행 바이너리·벤치마크 자산·외부 플래그 정답표는 재배포하지 않습니다. CyBench의 선택한 하위 질문·짧은 별칭과 과제 경로는 `data/task_manifest.json`에 출처와 함께 보존했습니다. 이 저장소의 게시로 각 자원의 원래 이용 조건이 바뀌는 것은 아닙니다. 각 제공처의 LICENSE·모델 카드·데이터 설명을 확인할 수 있도록 링크를 제공합니다.

Self-Consistency, Pass@K, Semantic Entropy, self-consistent errors, selective prediction의 관련 연구 서지는 [논문 PDF](CISC-W_2026_paper__324.pdf)의 참고문헌을 따릅니다. 공개 도구의 경험적 재집계를 원 연구의 새로운 구현·재현으로 표현하지 않습니다.
