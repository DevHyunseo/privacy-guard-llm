"""
KLUE-RoBERTa 개인정보 문맥 이해 테스트
"""

import torch
import time
import sys
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re

def test_klue_roberta_installation():
    """KLUE-RoBERTa 설치 확인"""
    try:
        from transformers import AutoTokenizer, AutoModel
        print("[성공] KLUE-RoBERTa (transformers) 라이브러리 import 성공!")
        return True
    except ImportError as e:
        print("[실패] KLUE-RoBERTa import 실패: {}".format(e))
        return False

def load_klue_roberta_model():
    """KLUE-RoBERTa 모델 로딩"""
    try:
        print("[로딩] KLUE-RoBERTa 모델 로딩 중...")
        start_time = time.time()

        model_name = "klue/roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        model.eval()
        load_time = time.time() - start_time

        print("[성공] KLUE-RoBERTa 모델 로딩 완료! (소요시간: {:.2f}초)".format(load_time))
        print("[정보] 모델: {}".format(model_name))
        print("[정보] 어휘 사전 크기: {}".format(tokenizer.vocab_size))

        return model, tokenizer
    except Exception as e:
        print("[실패] 모델 로딩 실패: {}".format(e))
        return None, None

def test_tokenization(tokenizer):
    print("\n[토큰] KLUE-RoBERTa 토크나이징 테스트")
    print("-" * 50)

    test_sentences = [
        "안녕하세요 김철수입니다",
        "전화번호는 010-1234-5678입니다",
        "35세 남성 의사이고 강남구에 거주합니다",
        "혈당 350, 케톤산증으로 응급실에 내원한 40대 남성",
        "API 키는 sk-abc123입니다"
    ]

    for sent in test_sentences:
        tokens = tokenizer.tokenize(sent)
        token_ids = tokenizer.encode(sent, add_special_tokens=True)
        print(f"원문: {sent}")
        print(f"토큰: {tokens}")
        print(f"토큰 수: {len(tokens)}, ID 수: {len(token_ids)}")
        print()

def get_sentence_embedding(model, tokenizer, text):
    """[CLS] 토큰 임베딩 추출"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

def test_similarity(model, tokenizer):
    print("\n[유사도] KLUE-RoBERTa 문맥 유사도 테스트")
    print("-" * 50)

    pairs = [
        ("의사선생님", "의사 선생님", "띄어쓰기 차이"),
        ("김철수씨", "김철수님", "호칭 차이"),
        ("혈당 350", "혈압 140/90", "의료 수치"),
        ("강남구 역삼동", "강남구역삼동", "주소 표기 차이"),
    ]

    for a, b, desc in pairs:
        emb1 = get_sentence_embedding(model, tokenizer, a)
        emb2 = get_sentence_embedding(model, tokenizer, b)
        sim = torch.cosine_similarity(emb1, emb2, dim=0).item()

        print(f"[{desc}]")
        print(f"  텍스트1: {a}")
        print(f"  텍스트2: {b}")
        print(f"  유사도: {sim:.4f}")
        if sim > 0.8:
            print("  [높음] 유사한 문맥 이해")
        elif sim > 0.6:
            print("  [중간] 어느 정도 유사")
        else:
            print("  [낮음] 문맥 차이 큼")
        print()

def test_privacy_detection(model, tokenizer):
    print("\n[개인정보] KLUE-RoBERTa 개인정보 감지 테스트")
    print("-" * 50)

    cases = [
        ("김철수입니다", "이름", "HIGH"),
        ("전화번호는 010-1234-5678입니다", "전화번호", "HIGH"),
        ("35세 남성 의사", "조합 정보", "MEDIUM"),
        ("혈당 350, 케톤산증 응급", "의료정보", "HIGH"),
        ("오늘 날씨가 좋네요", "일상", "NONE"),
    ]

    for text, category, expected in cases:
        emb = get_sentence_embedding(model, tokenizer, text)
        score = calculate_privacy_score(text)

        if score >= 0.8:
            risk = "CRITICAL"
        elif score >= 0.6:
            risk = "HIGH"
        elif score >= 0.4:
            risk = "MEDIUM"
        elif score >= 0.1:
            risk = "LOW"
        else:
            risk = "NONE"

        print(f"[{category}]")
        print(f"  텍스트: {text}")
        print(f"  예상 위험도: {expected}")
        print(f"  계산 점수: {score:.2f} → 판정: {risk}")
        print("  임베딩 차원: {}\n".format(emb.shape))

def calculate_privacy_score(text):
    score = 0.0
    if re.search(r'[가-힣]{2,4}(?=\s|님|씨|$)', text): score += 0.4
    if re.search(r'010-\d{4}-\d{4}', text): score += 0.4
    if re.search(r'\d{1,2}세|\d{1,2}살', text): score += 0.2
    if re.search(r'남성|여성|남자|여자', text): score += 0.1
    if re.search(r'의사|간호사|교수', text): score += 0.1
    if re.search(r'[가-힣]+구|[가-힣]+동', text): score += 0.1
    if re.search(r'환자|수술|혈당|혈압|응급', text): score += 0.2
    if re.search(r'sk-|API|키|password', text): score += 0.3
    return min(score, 1.0)

def test_performance(model, tokenizer):
    print("\n[성능] KLUE-RoBERTa 추론 성능 측정")
    print("-" * 50)

    texts = [
        "안녕하세요.",
        "김철수 교수님이 강의하셨습니다.",
        "혈당 수치가 350이고, 환자는 응급실에 내원했습니다. 주소는 강남구 역삼동이며 연락처는 010-1234-5678입니다."
    ]

    for i, text in enumerate(texts, 1):
        start = time.time()
        emb = get_sentence_embedding(model, tokenizer, text)
        end = time.time()

        elapsed = (end - start) * 1000
        print(f"[샘플 {i}] 길이: {len(text)}자 | 추론 시간: {elapsed:.2f}ms | 임베딩 크기: {emb.shape}")

def main():
    print("[시작] KLUE-RoBERTa 개인정보 감지 테스트")
    print("=" * 60)

    if not test_klue_roberta_installation():
        sys.exit(1)

    model, tokenizer = load_klue_roberta_model()
    if model is None or tokenizer is None:
        sys.exit(1)

    test_tokenization(tokenizer)
    test_similarity(model, tokenizer)
    test_privacy_detection(model, tokenizer)
    test_performance(model, tokenizer)

    print("\n[요약] KLUE-RoBERTa 테스트 완료")
    print("=" * 60)
    print("[성공] 기능 확인:")
    print("  --")
    print("  --")
    print("  --")
    print()
    print("[강점]")
    print("  -")
    print("  -")
    print()
    print("[한계]")
    print("  - ")
    print("  - ")

if __name__ == "__main__":
    main()