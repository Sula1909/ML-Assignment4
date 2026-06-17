# 😐 Facial Expression Recognition (FER) — Kaggle & PyTorch

## 📌 კონკურსის მიმოხილვა

[Kaggle FER Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/) — **Challenges in Representation Learning: Facial Expression Recognition Challenge**.

ამოცანა არის **7 კლასიანი კლასიფიკაცია** (სიბედი, სიბრაზე, სიზარალე, სიხარული, სიყვარული, შიში, შინაარსის არარსებობა) 48×48 ნაცრისფერი სახის სურათებზე. მონაცემები წარმოდგენილია `pixels` სვეტით (2304 პიქსელი) და `emotion` ლეიბლით.

**მთავარი მეტრიკა:** სიზუსტე (Accuracy) validation/test split-ზე.

## 🧠 ჩემი მიდგომა

მიზანი იყო **პრაქტიკული გამოცდილება ნეირონულ ქსოვილებში PyTorch-ით**, სხვადასხვა არქიტექტურის შედარება, ჰიპერპარამეტრების tuning და ყველა ექსპერიმენტის **Weights & Biases**-ზე ლოგირება.

მიდგომა იყო **იტერაციული**: პატარა Baseline CNN → უფრო ღრმა ქსელები → რეგულარიზაცია (BatchNorm, Dropout, Augmentation) → Attention მოდულები → CNN+Transformer ჰიბრიდები → **Ensemble**.

ლექტორის მოთხოვნის შესაბამისად, **ყველა გადაწყვეტილება ამ README-შია ახსნილი** — მხოლოდ საუკეთესო მოდელის კოდის წარდგენა არ არის საკმარისი.

**Wandb პროექტი:** [CNN](https://wandb.ai/gsula22-free-university-of-tbilisi-/CNN)

---

## 📁 რეპოზიტორიის სტრუქტურა

```
ML-Assignment4/
├── model_experiment_CNN.ipynb           ← მოდელები 1–4 (Baseline → Deep → Optimal → Attention)
├── model_experiment_CNN_Transformer.ipynb ← CNN+Transformer ჰიბრიდები + Ensemble
├── model_experiment_HP.ipynb            ← ჰიპერპარამეტრების მოკლე (15 ep) ძებნა
├── README.md
└── data/
    ├── example_submission.csv   ← რეპოზიტორიაშია
    ├── train.csv                ← Kaggle-დან ჩამოტვირთვა (არ არის GitHub-ზე — ძალიან დიდი)
    └── test.csv                 ← Kaggle-დან ჩამოტვირთვა
```

> **მონაცემები GitHub-ზე არ იტვირთება** (`train.csv` ~240MB, `icml_face_data.csv` ~300MB — GitHub-ის ლიმიტი 100MB). ჩამოტვირთე [Kaggle კონკურსიდან](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data) და დააყენე `data/` ფოლდერში. Kaggle ნოუთბუქში მონაცემები ავტომატურად ერთვება.

---

## 📄 ფაილების აღწერა

| ფაილი | დანიშნულება |
|-------|-------------|
| **model_experiment_CNN.ipynb** | ძირითადი CNN ევოლუცია: 4 მოდელის ოჯახი, sanity check, HP, wandb ლოგები |
| **model_experiment_CNN_Transformer.ipynb** | ViT/POSTER/EfficientFace ჰიბრიდები, checkpoint-ები, საბოლოო ensemble |
| **model_experiment_HP.ipynb** | ViT და POSTERLite-ის learning rate შედარება (15 ეპოქი); ensemble წონების ძებნა (val) |
| **data/train.csv** | საკონკურსო სატრენინგო მონაცემები (~28.7k სურათი) |
| **data/test.csv** | ტესტ მონაცემები ლეიბლების გარეშე (submission-ისთვის) |

---

## 🧼 მონაცემების მომზადება

### ➤ Dataset და split

- `FERDataset` — CSV-დან 48×48 გრეისკეილ ტენზორების აწყობა, ნორმალიზაცია [0, 1].
- **Split:** 70% train / 15% validation / 15% test, `random_seed=42` (ყველა ნოუთბუქში ერთნაირი).
- **Augmentation** (train-ზე): `RandomHorizontalFlip`, `RandomRotation(12°)`, `RandomAffine`.

### ➤ კლასების დისბალანსი

Train-ზე ემოციები არათანაბრადაა განაწილებული (მაგ. „სიბედი“ ყველაზე ხშირია). გამოვთვალე `CLASS_WEIGHTS`, მაგრამ ViT-ზე class-weighted loss **გაუარესდა** (~54% val) — ამიტომ საბოლოო მოდელებში **გამოვიყენე ჩვეულებრივი CrossEntropy**.

---

## 🔬 Forward / Backward გადამოწმება

ლექციაში განხილული სტრატეგიის შესაბამისად, სრული ტრენინგის წინ ვიყენებ `sanity_check()`-ს:

1. 1–2 სურათზე მოდელი **overfit**-ს აკეთებს (`loss.backward()` + `optimizer.step()`).
2. მოსალოდნელია loss → ~0 და accuracy → 100%.

ეს ამოწმებს, რომ **forward და backward გრაფი სწორადაა დაკავშირებული** და ოპტიმიზატორი მუშაობს. ნოუთბუქებში toggle-ითაა ჩართული (მაგ. `RUN_BASELINE_SANITY_CHECK`, `RUN_HYBRID_SANITY_CHECK`).

---

## 🧪 Training — იტერაციული არქიტექტურები

ქვემოთ მოცემულია თითოეული ეტაპის **რატომ**, **რა შედეგი** და **რა გავაკეთე შემდეგ**.

### მოდელი 1 — BaselineCNN (`model_experiment_CNN.ipynb`)

**რატომ:** პატარა არქიტექტურით დავიწყე — 2 conv ბლოკი, მინიმალური პარამეტრები.

**შედეგი:** val **46.24%**, test **45.41%**.

**გადაწყვეტილება:** სიღრმის გაზრდა საჭიროა; გადავედი DeepCNN-ზე.

---

### მოდელი 2 — DeepCNN (`model_experiment_CNN.ipynb`)

**რატომ:** უფრო ღრმა ქსელი უკეთ უნდა იჭერდეს რთულ ნიმუშებს.

**შედარებული ვარიანტები:**

| არქიტექტურა | Best val | Train acc | დასკვნა |
|-------------|----------|-----------|---------|
| **DeepCNN_Wide** | **52.58%** | **98.05%** | გამარჯვებული val-ზე |
| DeepCNN_WideFC | 50.65% | 97.48% | |
| DeepCNN_DeepNarrow | 49.93% | 96.63% | |

**Overfitting ანალიზი:** train ~98% vs val ~53% — კლასიკური **overfit** (არ არის BatchNorm/Dropout). ეს მნიშვნელოვანი მაგალითია README/ანგარიშისთვის.

**შედეგი:** DeepCNN_Wide test **50.06%**.

**გადაწყვეტილება:** რეგულარიზაცია საჭიროა → OptimalCNN.

---

### მოდელი 3 — OptimalCNN (`model_experiment_CNN.ipynb`)

**რატომ:** Deep მოდელის overfit-ის შესამცირებლად ეტაპობრივად დავამატე:

1. **BatchNorm** (dropout=0)
2. **BatchNorm + Dropout**
3. **BN + Dropout + Augmentation** (საბოლოო კონფიგი)

**შედარება (30 ep, val):**

| ვარიანტი | Best val |
|----------|----------|
| CNN_BatchNorm | **57.69%** |
| CNN_BatchNorm (40 ep) | 57.11% |

**გადაწყვეტილება:** BatchNorm მნიშვნელოვნად აუმჯობესებს განზოგადებას; შემდეგ ეტაპზე გადავედი Attention-ზე.

---

### მოდელი 4 — CNN + Attention (`model_experiment_CNN.ipynb`)

**რატომ:** სივრცითი attention (SE, CBAM, ECA) უნდა დაეხმარა მოდელს მნიშვნელოვან ფიჩერებზე ფოკუსირებაში.

**შედარება (40 ep, augmentation, BN+dropout):**

| მოდული | Best val |
|--------|----------|
| **SE** | **59.64%** |
| CBAM | 59.54% |
| ECA | 59.36% |

**გადაწყვეტილება:** SE გამარჯვებული; შემდეგ Transformer ჰიბრიდებზე გადავედი გლობალური კონტექსტისთვის.

---

### მოდელი 5 — CNN + Transformer ჰიბრიდები (`model_experiment_CNN_Transformer.ipynb`)

**რატომ:** FER-ში სივრცითი ურთიერთობები მნიშვნელოვანია; Transformer ტოკენებს შორის გლობალურ კონტექსტს იჭერს.

**შედარებული არქიტექტურები:**

| მოდელი | ინსპირაცია | Best val (შედარება) |
|--------|------------|---------------------|
| **CNN_ViT_Hybrid** | ViT-სტილი | **60.85%** (30 ep) |
| POSTERLite | POSTER pyramid | 60.64% (40 ep) |
| EfficientFaceLite | EfficientFace | 58.96% (40 ep) |

**საბოლოო ViT (40 ep, lr=1e-3):** val **61.22%**, test **59.23%**.

---

### მოდელი 6 — Ensemble (Path C)

**რატომ:** ViT და POSTERLite სხვადასხვა არქიტექტურაა — სხვადასხვა შეცდომებს აკეთებენ; **logit-ების საშუალო** ხშირად უკეთესია.

**მეთოდი:** `ensemble_logits = (ViT_logits + POSTER_logits) / 2`

**საუკეთესო შედეგი:**

| მოდელი | Test accuracy |
|--------|---------------|
| ViT მარტივად (lr=1e-3) | 59.23% |
| **Ensemble ViT + POSTERLite** | **62.15%** |

Wandb run: `Ensemble_ViT_POSTERLite_FinalTest` (test_accuracy = **0.62155**).

---

## ⚙️ Hyperparameter tuning

ჰიპერპარამეტრების ძებნა ცალკე ნოუთბუქშია (`model_experiment_HP.ipynb`) — **15 ეპოქიანი მოკლე გაშვებები**, val-ით შედარება (სრული 40 ep-ის ნაცვლად).

### BaselineCNN — learning rate (`model_experiment_CNN.ipynb`)

| LR | Best val |
|----|----------|
| **1e-3** | **46.24%** |
| 5e-4 | 43.89% |
| 1e-2 | 39.29% |

**დასკვნა:** `1e-2` ძალიან დიდია → **underfitting/არასტაბილური სწავლა**; `1e-3` ოპტიმალურია Baseline-ისთვის.

### CNN_ViT_Hybrid — learning rate (15 ep)

| LR | Best val @ 15 ep |
|----|------------------|
| **5e-4** | **57.20%** |
| 1e-3 | 55.41% |
| 2e-3 | 41.76% |

**დასკვნა:** `2e-3` უარყოფითი; `5e-4` სწრაფად სწავლობს, მაგრამ **40 ep სრულ ensemble-ზე საუკეთესო დარჩა `lr=1e-3`** (62.15% test).

### POSTERLite — learning rate (15 ep)

| LR | Best val @ 15 ep |
|----|------------------|
| **5e-4** | **57.29%** |
| 1e-3 | 56.83% |
| 2e-3 | 54.55% |

იგივე ტენდენცია, რაც ViT-ზე.

---

## 📉 Overfitting / Underfitting — მნიშვნელოვანი მაგალითები

ლექტორი ამბობს, რომ overfit/underfit მოდელების ანალიზი უფრო მნიშვნელოვანია, ვიდრე მხოლოდ მაღალი ქულა.

| მოვლენა | მაგალითი | მიზეზი |
|---------|----------|--------|
| **Overfitting** | DeepCNN: train ~98%, val ~53% | ღრმა ქსელი, არ არის BN/Dropout |
| **Underfitting (LR)** | Baseline `lr=1e-2`: val 39.29% | ზედმეტად დიდი learning rate |
| **Underfitting (LR)** | ViT/POSTER `lr=2e-3` @ 15 ep: ~42–54% val | ზედმეტად აგრესიული ნაბიჯი |
| **ცუდი სტრატეგია** | Class-weighted ViT: ~54% val | ძალით over-emphasis ნაკლებ კლასებზე |
| **Ensemble არ ყოველთვის უკეთესი** | lr5e-4 ViT + POSTER: ensemble 61.16% < ViT alone 61.25% | მოდელები ძალიან მსგავს შეცდომებს აკეთებენ |

---

## 📊 Wandb Tracking

**პროექტი:** [CNN](https://wandb.ai/gsula22-free-university-of-tbilisi-/CNN)

MLflow-ის მსგავსად, **თითოეული არქიტექტურა/ექსპერიმენტი ცალკე run-ია**, მაგალითად:

- `Baseline_CNN`, `Baseline_CNN_HyperparameterSelection_lr*`
- `DeepCNN_Wide`, `DeepCNN_WideFC`, `DeepCNN_DeepNarrow`
- `CNN_BatchNorm`, `OptimalCNN_*`
- `SE`, `CBAM`, `ECA`
- `CNN_ViT_Hybrid`, `POSTERLite`, `EfficientFaceLite`
- `CNN_ViT_Hybrid_Final`, `POSTERLite_EnsembleCkpt`
- `ViT_HP_lr5e-4`, `ViT_HP_lr2e-3`, `POSTER_HP_lr5e-4`, `POSTER_HP_lr2e-3`
- `Ensemble_ViT_POSTERLite_FinalTest`

თითო run-ში ლოგირდება: `epoch`, `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `best_val_accuracy`, კონფიგი (`learning_rate`, `batch_size`, `model`, და ა.შ.).

**ბონუსი:** Wandb Report-ის შექმნა რეკომენდებულია (ჯგუფები: CNN ევოლუცია / Hybrid / HP / Ensemble).

---

## 🏆 საბოლოო მოდელის შერჩევა

**საბოლოო მოდელი:** **Ensemble — CNN_ViT_Hybrid + POSTERLite** (equal-weight logit average)

| მეტრიკა | მნიშვნელობა |
|---------|-------------|
| Learning rate (ორივე) | **1e-3** |
| ViT val (40 ep) | 61.22% |
| ViT test | 59.23% |
| POSTERLite val | 62.01% |
| **Ensemble test** | **62.15%** |

**რატომ ensemble და არა მარტო ViT:** ensemble +2.9 პუნქტით აღემატება ViT-ს test-ზე; POSTERLite სხვა არქიტექტურული ხედვაა (multi-scale pyramid).

**რატომ არა `lr=5e-4`:** 15-ep HP-ში უკეთესი იყო, მაგრამ 40-ep + ensemble სესიაში `1e-3` მოგვცა უმაღლესი ensemble test.

Checkpoint-ები (იმავე Kaggle სესიაში):  
`/kaggle/working/checkpoints/CNN_ViT_Hybrid.pt` და `POSTERLite.pt`  
→ **Save Version → Save output** რეკომენდებულია.

---

## 📈 მეტრიკების განმარტება

- **Validation accuracy** — HP შედარებისა და early stopping-ისთვის; test-ზე ერთხელ ვაფასებთ საბოლოოს.
- **Test accuracy** — held-out 15% split (seed=42); საბოლოო რიცხვი ანგარიშისთვის.
- **Train vs val gap** — overfitting-ის ინდიკატორი (განსაკუთრებით DeepCNN-ზე).

---

## ▶️ როგორ გავუშვათ (Kaggle)

1. **CNN ევოლუცია:** `model_experiment_CNN.ipynb` — toggle-ები თითო მოდელისთვის (`RUN_* = False` დასრულებული ექსპერიმენტებისთვის).
2. **Transformer + Ensemble:** `model_experiment_CNN_Transformer.ipynb`  
   - საბოლოო ensemble უკვე ჩაწერილია; training toggle-ები **False** (არ გაუშვათ Run All შემთხვევით).
3. **HP:** `model_experiment_HP.ipynb` — `RUN_VIT_LR_SEARCH` / `RUN_POSTER_LR_SEARCH` (დასრულებული); ensemble weight search საჭიროებს checkpoint-ებს.
4. **Wandb:** `WANDB_API_KEY` Kaggle Secrets-ში.

---

## 🔗 ბმულები

- **Wandb პროექტი:** https://wandb.ai/gsula22-free-university-of-tbilisi-/CNN
- **Kaggle კონკურსი:** https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/

---

## 📝 შეჯამება

| კრიტერიუმი | რა გავაკეთე |
|------------|-------------|
| **≥3 არქიტექტურა** | Baseline, Deep (3), Optimal, Attention (3), Hybrid (3), Ensemble |
| **იტერაციული განვითარება** | ფენობრივად დავამატე სიღრმე, რეგულარიზაცია, attention, transformer |
| **HP tuning** | LR Baseline, ViT, POSTER; არქიტექტურული შედარებები |
| **Forward/backward** | `sanity_check()` ყველა ოჯახში |
| **Over/underfit ანალიზი** | DeepCNN overfit, LR ექსტრემუმები, class weights ჩავარდნა |
| **Wandb** | ცალკე run თითო ექსპერიმენტზე |
| **საუკეთესო შედეგი** | Ensemble test **62.15%** |
