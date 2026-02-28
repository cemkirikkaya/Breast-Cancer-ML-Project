# Meme Kanseri Wisconsin Veri Seti – İkili Sınıflandırma

Bu proje, **Breast Cancer Wisconsin Dataset** kullanılarak iyi huylu (benign) ve kötü huylu (malignant) tümörlerin makine öğrenmesi ile sınıflandırılmasını içerir.

## Veri seti

- **Özellikler (30 adet):** Her biri için ortalama (mean), standart hata (standard error) ve en kötü değer (worst) hesaplanmıştır.
  - Radius (Yarıçap), Texture (Doku), Perimeter (Çevre), Area (Alan), Smoothness (Düzgünlük), Compactness, Concavity, Symmetry, Fractal Dimension
- **Sınıf etiketi:** İkili (benign / malignant).

## Kullanım

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. İsteğe bağlı: Veri seti CSV dosyanızı proje klasörüne koyun. Desteklenen dosya adları: `data.csv`, `breast_cancer.csv`, `breast_cancer_wisconsin.csv`, `wdbc.csv`. Sınıf sütunu `diagnosis` (M/B) veya benzeri olabilir; 30 özellik sütunu kullanılır. CSV yoksa script otomatik olarak scikit-learn içindeki Wisconsin veri setini kullanır.
3. Sınıflandırmayı çalıştırın:
   ```bash
   python meme_kanseri_siniflandirma.py
   ```

## Model ve değerlendirme

- **Yöntem:** Random Forest sınıflandırıcı (ikili sınıflandırma için uygun).
- **Metrikler:** Accuracy (Doğruluk), Precision (Kesinlik), Recall (Duyarlılık), F1-Score.
- Veri %80 eğitim, %20 test olarak ayrılır; özellikler standartlaştırılır.

## Çıktı

Program, test seti üzerinde Doğruluk, Kesinlik, Duyarlılık ve F1-Score değerlerini, karışıklık matrisini ve sınıflandırma raporunu Türkçe etiketlerle yazdırır.
