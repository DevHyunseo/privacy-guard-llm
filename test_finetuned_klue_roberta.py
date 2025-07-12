import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
import datetime
import os
import glob
import re

def predict_privacy_risk(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=-1).item()
    return pred

def parse_model_info(model_dir):
    match = re.search(r'ep(\d+)_lr([0-9.eE-]+)_bs(\d+)', model_dir)
    if match:
        epochs = int(match.group(1))
        lr = float(match.group(2))
        batch_size = int(match.group(3))
        return epochs, lr, batch_size
    else:
        return None, None, None

def main():
    result_dir = "klue_roberta_result"
    os.makedirs(result_dir, exist_ok=True)

    # 자동으로 모델 디렉토리 리스트업
    model_dirs = sorted(glob.glob("./finetuned_klue_roberta_privacy_ep*"))

    id2label = {0: 'NONE', 1: 'LOW', 2: 'MEDIUM', 3: 'HIGH', 4: 'CRITICAL'}
    label2id = {v: k for k, v in id2label.items()}

    test_data = [
        ("김철수 환자 혈당 350, 응급실 내원", "HIGH"),
        ("오늘 날씨가 좋네요.", "NONE"),
        ("환자번호 P20240101, 박영희(42세 여성), 고혈압으로 입원", "CRITICAL"),
        ("서울대병원 응급실은 24시간 운영됩니다.", "LOW"),
        ("임신 32주, 산모 나이 28세, 정기검진 이상무", "HIGH"),
        ("고혈압 환자는 저염식을 하는 것이 좋습니다.", "LOW"),
        ("40대 여성, 유방암 가족력, 서울대병원 외래 진료", "HIGH"),
        ("혈압 180/100, 콜레스테롤 280으로 고위험군", "MEDIUM"),
        ("간호사 박영희가 환자 케어를 담당하고 있습니다.", "MEDIUM"),
        ("당뇨병은 혈당 조절이 중요한 질환입니다.", "LOW"),
        ("35세 남성 환자 김철수씨가 당뇨 진단받고 메트포르민 처방받았습니다.", "HIGH"),
        ("김철수 의사선생님이 오늘 수술 집도하셨습니다.", "MEDIUM"),
        ("분당서울대병원에 방문한 김철수(60세 여성)은 폐렴로 치료 받음.", "HIGH"),
        ("API 키는 sk-abc123입니다", "CRITICAL"),
        ("혈당 350mg/dL, 케톤산증 의심으로 응급실 내원", "MEDIUM"),
    ]

    summary_path = os.path.join(result_dir, "test_results_summary.csv")
    all_summary_rows = []

    for model_dir in model_dirs:
        epochs, lr, batch_size = parse_model_info(model_dir)
        if epochs is None:
            print(f"[경고] 모델 경로에서 튜닝 정보를 파싱할 수 없습니다: {model_dir}")
            continue

        print(f"\n[평가] {model_dir} (epochs={epochs}, lr={lr}, batch_size={batch_size})")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        y_true = []
        y_pred = []
        results = []

        for sent, true_label in test_data:
            pred_id = predict_privacy_risk(sent, model, tokenizer)
            pred_label = id2label[pred_id]
            y_true.append(label2id[true_label])
            y_pred.append(pred_id)
            results.append({
                "문장": sent,
                "실제 위험도": true_label,
                "예측 위험도": pred_label,
                "모델": os.path.basename(model_dir)
            })

        # 문장별 결과 저장 (모델별로 파일명 다르게)
        df_detail = pd.DataFrame(results)
        detail_path = os.path.join(result_dir, f"test_results_detail_{os.path.basename(model_dir)}.csv")
        df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

        # 성능 평가 및 summary 누적
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        _, _, f2, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', beta=2, zero_division=0)
        acc = (pd.Series(y_true) == pd.Series(y_pred)).mean()

        summary_row = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": "klue/roberta-base",
            "augment_per_class": 200,
            "Accuracy": round(acc, 4),
            "Macro Precision": round(precision, 4),
            "Macro Recall": round(recall, 4),
            "Macro F1": round(f1, 4),
            "Macro F2": round(f2, 4)
        }
        all_summary_rows.append(summary_row)

        print(f"[저장] {detail_path} 저장 완료.")

    # summary 파일 누적 저장
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = pd.concat([df_summary, pd.DataFrame(all_summary_rows)], ignore_index=True)
    else:
        df_summary = pd.DataFrame(all_summary_rows)
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] 모든 실험 요약이 {summary_path}에 누적 저장되었습니다.")

if __name__ == "__main__":
    main()