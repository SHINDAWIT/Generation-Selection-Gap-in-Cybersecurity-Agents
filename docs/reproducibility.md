# 공개 자료 검증과 실험 재현 범위

## 공개 저장소만으로 실행하는 검증

Python 3.11 이상, 표준 라이브러리만 사용합니다. 저장소 루트에서 실행합니다.

```bash
python3 -B scripts/verify_public_results.py
PYTHONPATH=src:. python3 -B -m unittest discover -s tests -v
```

첫 명령은 다음을 검사합니다.

1. 게시 파일의 SHA-256·크기와 Python 소스의 문법.
2. 개발 15×50, 미사용 11×20, 외부 35×20의 완전한 과제·시드 격자.
3. 동일 과제에서 같은 답 식별자에 다른 정오·의미 라벨이 붙지 않았는지.
4. 개발 K=5/25의 strict Oracle·최빈 정답 수.
5. 개발 K=50의 별도 사후 의미 정오·Oracle 수와 MOTP 49:1 사례.
6. K=20의 기권 전후 정답·오답·기권 수와 무효 실행 수.

테스트는 원 연구 코드의 선택·정규화·도구·실행 경계 37개와, 공개용 기록의 누락·중복·라벨 불일치·파일 변조 및 선택 경계 7개로 구성됩니다. 공개 시점의 테스트 실행과 과거 연구 당시의 테스트 실행을 구분합니다.

## 공개용 기록의 의미

[rollout_answers.jsonl](../data/public/rollout_answers.jsonl)은 원시 대화 전체가 아니라 기존 모델 실행에서 필요한 필드를 추린 기록입니다.

| 필드 | 내용 |
| --- | --- |
| `cohort`, `task_id`, `source` | 평가 집합, 부모 과제 ID, 출처 |
| `seed` | 실제 기록의 독립 실행 시드 |
| `answer_key` | 유효한 정규화 답의 SHA-256. 무효면 `null` |
| `strict_correct` | 원 정답 별칭 대조에서 얻은 정오. 무효는 `false` |
| `semantic_cluster`, `semantic_label` | 개발 로그의 사후 단일 주석. K=20에는 `null` |

식별자는 답의 동일성·빈도를 유지하므로 최빈 선택과 기권을 다시 계산할 수 있습니다. SHA-256 처리는 익명성이나 비밀성을 보장하는 수단으로 제시하지 않습니다. 원문 대화·추론문·도구 반환 본문과 외부 최종 플래그 정답표는 배포하지 않습니다. CyBench 개발 하위 질문의 별칭과 의미 주석은 [과제 manifest](../data/task_manifest.json)와 [군집 규칙](../data/annotations/k50_semantic_clusters.json)에 있습니다.

**재집계의 범위**: 정오 라벨과 의미 군집은 기존 분석에서 가져옵니다. 공개 검증은 그 라벨의 원문 기반 타당성, 사후 주석의 독립성, 원시 자료에서 공개 기록으로의 변환 전체를 독립적으로 입증하지 않습니다. 원시 입력 식별용 해시는 [source_inputs.json](../data/public/source_inputs.json)에 있으며, 원시 파일 자체는 포함하지 않습니다.

부트스트랩 구간, 의미 AUROC, 전체 비용·도구 추적은 이 공개용 도구의 재계산 대상이 아닙니다. 보존 요약의 값은 [결과 문서](results.md)에서 구분해 설명합니다.

## 새 모델 실행을 준비하는 경우

아래는 환경 준비 후의 실행 예시이며, 이번 게시 과정에서 새로운 GPU 실험을 수행하지 않았습니다.

1. [CyBench](https://github.com/andyzorigin/cybench)의 commit `1097a7226eb034d3821208114da38f10b8627ab1`을 별도 경로에 준비합니다.
2. [Granite GGUF](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF)에서 해당 Q4_K_M 모델과 [llama.cpp b11026](https://github.com/ggml-org/llama.cpp/releases/tag/b11026)의 적절한 CUDA 실행 환경을 준비합니다.
3. 서버 시작 스크립트가 찾는 `.tools/llama.cpp/` 아래에 `llama-server`와 `libcudart.so.12`, `models/` 아래에 모델을 배치하거나 `MODEL_PATH`를 설정합니다. 이 바이너리·가중치는 저장소에 포함하지 않습니다.
4. 과제 준비 후 실제 경로·모델·추론 설정을 확인하고 별도 새 로그에 실행합니다.

```bash
python3 scripts/prepare_cybench_subset.py --cybench-root /path/to/cybench
bash scripts/start_llama_server.sh
```

서버가 준비된 별도 터미널에서 현재 prompt v2 실행기 예시:

```bash
PYTHONPATH=src:. python3 scripts/run_experiment.py \
  --split pilot --output data/raw/publication_v2_pilot.jsonl
```

**이 명령은 과거 prompt v1 개발 실험과 동일한 재실행이 아닙니다.** 현재 실행기는 prompt v2를 기본으로, `--artifact-path-guidance`에서 v3를 사용합니다. 새 결과를 보존된 K=5/25/50 결과와 합치지 않습니다. 실제 준비 과정의 성공이나 GPU 추론 결과는 이번 공개 검증 범위에 포함되지 않습니다.

## 보존 분석기의 입력과 제약

`analyze_k50.py`, `analyze_k25.py`, `analyze_semantic_entropy.py`, `analyze_holdout.py`, `analyze_external_transfer.py`는 연구 코드 검토와 원시 자료 보유 환경을 위한 스냅샷입니다. 해당 스크립트의 기본 경로에는 배포하지 않은 raw JSONL·준비 데이터·정답표가 포함되므로, 공개 저장소만으로 기본 실행을 완료할 수 없습니다. 특히 K=25 provenance 생성은 모델 파일과 런타임 해시도 읽습니다.

연구 폴더에서 K=25 검증은 수치·결정·표·그림의 일치 이후 코드 해시 차이를 감지했습니다. 그 사실을 [버전 감사 파일](../results/reference/k25_code_version_audit.json)에 보존했습니다. 기록을 현재 코드 해시로 덮어써 과거와 일치하는 것처럼 만들지 않았습니다.

## 큐레이터용 공개 기록 생성

원시 입력이 있는 연구 폴더를 별도로 보유한 경우에만 사용합니다.

```bash
python3 -B scripts/export_public_data.py --source-root /path/to/research-workspace
```

이 명령은 공개 저장소의 투영 데이터와 manifest를 갱신합니다. 보존 연구 폴더를 수정하거나 모델을 호출하지 않습니다. 데이터 변경 없이 문서·소스 파일의 게시 해시만 갱신하려면 `--manifest-only`를 사용합니다. 공개 검증 도구는 manifest의 자기 해시·Git 커밋의 외부 진위를 증명하는 도구가 아니라, 명시된 파일의 무결성을 확인하는 도구입니다.
