# 자료 출처와 공개용 변경

## 논문 기준

- 제목: 「사이버보안 에이전트의 반복 샘플링에서 정답 생성과 최빈 답 선택의 간극」.
- 영어 제목: The Gap Between Correct-Answer Generation and Modal Answer Selection in Multi-Rollout Cybersecurity Agents.
- 공개 파일: [CISC-W_2026_paper__324.pdf](../CISC-W_2026_paper__324.pdf).
- 입력 파일 식별: `cursor-to-WSL/CISC-W_2026_paper__324 (2).pdf`, 4쪽.
- PDF SHA-256: `27ecde4be3da1c497d04bb014d73c482e04b47960625d2a5d38efb6fb142cf2e`.

입력 PDF의 바이트는 유지하고 공개 파일명만 정리했습니다. 기존 `(1)` PDF와의 텍스트 대조에서 확인된 차이는 사사문 추가입니다. 파일명·게시일만으로 최종 제출·채택·발표 상태를 확정하지 않습니다. 저자·소속은 PDF의 표기를 따릅니다.

## 그대로 보존한 자료

선택한 `src/cisc_rollout/` 소스, 실행·준비·분석 스크립트, 원 테스트 37개, 과제 manifest, 개발 의미 군집 규칙, 원 프로토콜과 결과 요약은 현재 연구 폴더의 바이트를 유지했습니다. 파일별 `preserved_snapshot` 여부와 원 경로·해시는 [publication_manifest.json](../publication_manifest.json)에 있습니다.

이는 **현재 보존 소스의 스냅샷**을 뜻합니다. 모든 파일이 과거 실행 시점과 동일한 버전이라는 의미는 아닙니다. 원 문서의 연구 단계 표현과 현재 공개 설명이 다르면, 해당 문서의 시점·단계를 구분하고 [방법](methodology.md)의 적용 범위를 참고하십시오.

## 공개를 위해 새로 만든 자료

- 한국어·영어 README, 프로젝트 소개, 코드 안내, 방법·결과·재현·출처 문서, 인용 안내.
- 보존된 원 집계에서 필요한 정수만 추린 `results/paper_results.json`.
- 1,670개 실제 실행에서 답 식별자와 채점 라벨을 추린 `data/public/` 자료.
- 공개용 export·검증 도구, 누락·중복·파일 변조·정책 경계 등에 대한 테스트 7개.
- README 결과 그림. 원 PDF 그림을 수정한 것이 아니라 원 집계로 새로 그린 설명용 그림입니다.
- 게시 파일 해시 목록과 과거 K=25 코드 버전 감사.

이는 새로운 모델 실행이나 독립적인 재채점 연구 결과가 아닙니다. 공개 데이터의 단순 재집계와 기존 연구 수치·라벨의 독립적 검증은 [재현 문서](reproducibility.md)에서 구분합니다.

## 코드 버전 차이

1. 보존된 개발 K=50의 750개 기록은 모두 `static-cybench-json-tools-v1`입니다. 현재 `run_experiment.py`의 기본값은 `static-cybench-json-tools-v2-no-final-placeholder`이며, 경로 안내 옵션은 v3입니다.
2. 과거 `results/k25_provenance.json`에 기록된 해시와 현재 `scripts/analyze_k25.py`, `scripts/run_experiment.py`, `src/cisc_rollout/client.py`의 해시가 다릅니다. 이후 추가된 모듈도 있습니다.
3. 원시 연구 폴더에서 K=25의 수치·결정·CSV·요약·그림 대조는 통과했지만, 마지막 검사에서 `stored K=25 provenance mismatch: harness_sha256`로 종료했습니다.

[기록 해시와 현재 해시의 감사 결과](../results/reference/k25_code_version_audit.json)를 포함했습니다. 과거 provenance를 새 해시로 교체하거나 현재 소스를 역사적 실행 코드로 표시하지 않았습니다. 따라서 공개본은 핵심 결과 재집계와 코드 검토를 지원하지만, 모든 역사적 모델 실행의 동일 재생을 입증하지 않습니다.

## 포함하지 않은 자료

원시 모델 대화·추론문·도구 반환 본문, 외부 최종 플래그 정답표, 모델 가중치, llama.cpp 바이너리, 복제한 벤치마크 자산, 로컬 로그·런타임 상태, 심사 의견·답변 및 개인 전달 메모는 포함하지 않았습니다. 경로 안내·근거 검증기·선택 인터페이스·다중 모델·대화형 후속 실험도 이 저장소의 핵심 결과에 합치지 않았습니다.

개발 CyBench 하위 질문의 짧은 정답 별칭과 사후 의미 군집 규칙은 코드·채점 방법 검토를 위해 포함했습니다. 모든 정답 정보가 제거된 자료라고 표현하지 않습니다.

## 공동연구와 기존 도구

이 저장소의 Git 게시 이력은 공개 정리 이력입니다. 저자별 과거 구현·실험 기여를 소급해 증명하지 않습니다. [저자 정보](../CITATION.md)와 [기존 모델·벤치마크 출처](../THIRD_PARTY_NOTICES.md)를 함께 제공합니다. 원 자료에서 확인되지 않은 포괄적 오픈소스 라이선스를 새로 부여하지 않았습니다.
