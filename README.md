# Fashion-MNIST CNN Classifier

CNN بسيط بـ TensorFlow/Keras لتصنيف صور الملابس (10 classes، صور 28x28 grayscale).

## المميزات
- 3 conv blocks مع BatchNormalization و Dropout
- Data augmentation (RandomFlip + RandomTranslation)
- EarlyStopping و ReduceLROnPlateau
- حفظ كل النتائج (الرسومات والتقرير) في مجلد `results/`

## التشغيل
```bash
python -m venv .venv
source .venv/bin/activate        # على ويندوز: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

## النتائج
بعد التدريب هتلاقي في `results/`:
- `training_curves.png`
- `confusion_matrix.png`
- `sample_predictions.png`
- `classification_report.txt`

بعد ما تشغّله، ضيف هنا الدقة النهائية وصورة الـ confusion matrix:

```markdown
![Confusion Matrix](results/confusion_matrix.png)
```
