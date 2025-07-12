import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import random


def augment_privacy_cases(n_per_class=200):
    names = ["김철수", "박영희", "이민수", "최유리", "정하늘", "홍길동", "이수민"]
    ages = ["35세", "42세", "50세", "60세", "28세", "40대", "50대", "70세"]
    genders = ["남성", "여성"]
    hospitals = ["서울대병원", "분당서울대병원", "대구파티마병원", "부산의료원", "삼성서울병원"]
    diseases = ["당뇨", "고혈압", "유방암", "간암", "치매", "심근경색", "폐렴", "천식", "알츠하이머"]
    numbers = ["350", "180/100", "280", "140/90", "400", "120/80"]
    apikeys = ["sk-abc123", "sk-xyz789", "sk-123xyz", "sk-999999"]
    none_sentences = [
        "오늘 날씨가 맑습니다.",
        "점심에 파스타를 먹었습니다.",
        "주말에 산책을 했어요.",
        "책을 읽는 것이 취미입니다.",
        "커피를 마시며 휴식을 취했습니다.",
        "영화관에서 영화를 봤어요.",
        "친구와 저녁을 먹었습니다.",
        "운동을 하고 왔어요."
    ]

    cases = []

    # HIGH/CRITICAL
    for _ in range(n_per_class):
        name = random.choice(names)
        age = random.choice(ages)
        gender = random.choice(genders)
        hospital = random.choice(hospitals)
        disease = random.choice(diseases)
        num = random.choice(numbers)
        apikey = random.choice(apikeys)
        # HIGH
        cases.append({'text': f"{age} {gender} 환자 {name}가 {disease} 진단받고 {hospital}에서 치료받았습니다.", 'expected_risk': 'HIGH'})
        cases.append({'text': f"{hospital}에 방문한 {name}({age} {gender})은 {disease}로 치료 받음.", 'expected_risk': 'HIGH'})
        cases.append({'text': f"{name}({age} {gender})이 {hospital}에서 {disease} 수술을 받았습니다.", 'expected_risk': 'HIGH'})
        # CRITICAL
        cases.append({'text': f"환자번호 P2024{random.randint(1000,9999)}, {name}({age} {gender}), {disease}으로 입원", 'expected_risk': 'CRITICAL'})
        cases.append({'text': f"API 키는 {apikey}입니다", 'expected_risk': 'CRITICAL'})
        cases.append({'text': f"{name}의 주민등록번호는 900101-1234567입니다.", 'expected_risk': 'CRITICAL'})

    # MEDIUM/LOW
    for _ in range(n_per_class):
        disease = random.choice(diseases)
        num = random.choice(numbers)
        hospital = random.choice(hospitals)
        # MEDIUM
        cases.append({'text': f"{disease} 수치가 {num}입니다.", 'expected_risk': 'MEDIUM'})
        cases.append({'text': f"{disease} 환자는 정기검진이 필요합니다.", 'expected_risk': 'MEDIUM'})
        cases.append({'text': f"{hospital}에서 {disease} 관련 건강강좌가 열립니다.", 'expected_risk': 'MEDIUM'})
        # LOW
        cases.append({'text': f"{disease}은 건강 관리가 중요합니다.", 'expected_risk': 'LOW'})
        cases.append({'text': f"{hospital} 응급실은 24시간 운영됩니다.", 'expected_risk': 'LOW'})
        cases.append({'text': f"{disease} 예방을 위해 운동이 필요합니다.", 'expected_risk': 'LOW'})

    # NONE
    for _ in range(n_per_class):
        cases.append({'text': random.choice(none_sentences), 'expected_risk': 'NONE'})

    random.shuffle(cases)
    return pd.DataFrame(cases)

def encode_labels(df):
    label2id = {'NONE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
    df['label'] = df['expected_risk'].map(label2id)
    return df, label2id

def load_tokenizer_and_model(model_name, num_labels, label2id):
    print("[로딩] KLUE-RoBERTa 토크나이저/분류모델 로딩 중...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label={v: k for k, v in label2id.items()}
    )
    load_time = time.time() - start_time
    print(f"[성공] 모델/토크나이저 로딩 완료! (소요시간: {load_time:.2f}초)")
    print(f"[정보] 모델: {model_name}")
    print(f"[정보] 어휘 사전 크기: {tokenizer.vocab_size}")
    return model, tokenizer

class PrivacyDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1": f1}

def main():
    print("[시작] KLUE-RoBERTa 개인정보 위험도 분류 파인튜닝")
    print("=" * 60)

    df = augment_privacy_cases(n_per_class=200)
    df, label2id = encode_labels(df)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"[데이터] 학습: {len(train_df)}개, 검증: {len(val_df)}개")

    epoch_list = [20, 30, 40]
    lr_list = [2e-5, 3e-5, 5e-5]
    batch_list = [16, 32]

    for num_train_epochs in epoch_list:
        for learning_rate in lr_list:
            for batch_size in batch_list:
                print(f"\n[실험] epochs={num_train_epochs}, lr={learning_rate}, batch_size={batch_size}")

                model_name = "klue/roberta-base"
                model, tokenizer = load_tokenizer_and_model(model_name, num_labels=len(label2id), label2id=label2id)

                train_dataset = PrivacyDataset(train_df, tokenizer)
                val_dataset = PrivacyDataset(val_df, tokenizer)

                training_args = TrainingArguments(
                    output_dir=f'./results/ep{num_train_epochs}_lr{learning_rate}_bs{batch_size}',
                    num_train_epochs=num_train_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    learning_rate=learning_rate,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    logging_dir='./logs',
                    logging_steps=10,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1",
                    greater_is_better=True,
                    save_total_limit=2,
                    report_to="none"
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=compute_metrics,
                    tokenizer=tokenizer
                )

                print("[학습] 파인튜닝 시작")
                trainer.train()
                print("[학습] 파인튜닝 완료")

                save_path = f"./finetuned_klue_roberta_privacy_ep{num_train_epochs}_lr{learning_rate}_bs{batch_size}"
                trainer.save_model(save_path)
                print(f"[저장] 파인튜닝 모델이 {save_path} 에 저장되었습니다.")

if __name__ == "__main__":
    main()