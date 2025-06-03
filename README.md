# Merhaba ben Furkan Çolak 
# Human Face Recognition Project

This project focuses on [your project description here, e.g., training a model to recognize human facial features using deep learning].

## 📁 Dataset

This project uses the [Human Faces Dataset](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset) created by [Kaustubh Dhote](https://www.kaggle.com/kaustubhdhote), which is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

**You must provide attribution** to the original dataset creator and **any derivative work must be shared under the same license**.

---

## 🔍 Veri Seti (Türkçe)

Bu projede kullanılan veri seti: [Human Faces Dataset](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset), [Kaustubh Dhote](https://www.kaggle.com/kaustubhdhote) tarafından oluşturulmuş ve [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) lisansı ile paylaşılmıştır.



## 📌 License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license. See the [LICENSE](LICENSE) file for more information.



# Paralel Görüntü İşleme

Bu proje, CPU çekirdeklerini kullanarak görüntü işleme işlemlerini paralel olarak gerçekleştiren bir Python uygulamasıdır.

## Özellikler

- Çoklu CPU çekirdeği desteği
- Paralel görüntü işleme
- Desteklenen işlemler:
  - Bulanıklaştırma (Blur)
  - Kenar tespiti (Edge Detection)
  - Gri tonlama (Grayscale)

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```
## Kullanım

Programı çalıştırmak için:
```bash
python app.py
```

Program, test görüntüsüne üç farklı işlem uygulayacak ve sonuçları kaydedecektir:
- `output_blur.jpg`
- `output_edge.jpg`
- `output_grayscale.jpg`

## Nasıl Çalışır?

1. Program, mevcut CPU çekirdek sayısını tespit eder
2. Görüntü, çekirdek sayısı kadar parçaya bölünür
3. Her parça ayrı bir CPU çekirdeğinde işlenir
4. İşlenmiş parçalar birleştirilir ve sonuç kaydedilir

## Performans

Program, işlem süresini ve kullanılan CPU çekirdek sayısını ekrana yazdırır. Bu sayede paralel işlemenin performansını gözlemleyebilirsiniz. 