# قواعد مشروع المطبخ (GhostChef AI)

## الهدف

بناء مشروع احترافي للتنبؤ بالطلب وتقليل الهدر وزيادة الأرباح.

## Architecture

``` text
GhostChef_AI/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
├── models/
├── dashboard/
├── reports/
├── docs/
├── tests/
└── deployment/
```

## قواعد المشروع

1.  فهم الـBusiness قبل الكود.
2.  عدم تخطي المراحل.
3.  Notebook لكل مرحلة.
4.  عدم تعديل data/raw.
5.  حفظ البيانات المعالجة في processed.
6.  نقل الدوال المستقرة إلى src.
7.  حفظ النماذج داخل models.
8.  توثيق القرارات في docs/design_decisions.md.
9.  الفصل بين EDA وPreprocessing وFeature Engineering.
10. الحفاظ على الـArchitecture طوال المشروع.

## Workflow

1.  Business Understanding
2.  Business Requirements
3.  ML Problem Definition
4.  Data Understanding
5.  Data Schema & Merge Strategy
6.  Project Architecture
7.  EDA
8.  Preprocessing
9.  Feature Engineering
10. Modeling
11. Evaluation
12. Dashboard
13. Deployment
