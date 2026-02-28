#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meme Kanseri (Breast Cancer Wisconsin) Veri Seti - İkili Sınıflandırma
İyi huylu (benign) ve kötü huylu (malignant) tümör ayrımı.
Metrikler: Accuracy, Precision, Recall, F1-Score (Türkçe çıktılar).
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Veri seti dosya yolları
VERI_KLASORU = "veri"
VERI_DOSYALARI = [
    "data.csv",
    "breast_cancer.csv",
    "breast_cancer_wisconsin.csv",
    "wdbc.csv",
]


def veri_setini_yukle():
    """
    Önce proje klasöründe ve veri/ klasöründe CSV arar; bulamazsa
    scikit-learn içindeki Breast Cancer Wisconsin veri setini kullanır.
    """
    proje_klasoru = os.path.dirname(os.path.abspath(__file__))
    veri_klasoru_yolu = os.path.join(proje_klasoru, VERI_KLASORU)
    # Önce veri/ klasörüne bak (veri seti buraya yüklenir)
    aranacak_yerler = [veri_klasoru_yolu, proje_klasoru]

    for klasor in aranacak_yerler:
        for dosya_adı in VERI_DOSYALARI:
            yol = os.path.join(klasor, dosya_adı)
            if os.path.isfile(yol):
                print(f"Veri seti CSV'den yükleniyor: {yol}\n")
                return csv_den_yukle(yol)

    print("Proje klasöründe CSV bulunamadı. Sklearn Wisconsin veri seti kullanılıyor.\n")
    return sklearn_veri_seti_yukle()


def csv_den_yukle(dosya_yolu):
    """CSV dosyasından veri yükler (veri/data.csv formatı). Sınıf: diagnosis (M/B), 30 özellik."""
    import pandas as pd
    df = pd.read_csv(dosya_yolu, encoding="utf-8")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)  # BOM temizle

    # Wisconsin formatı: 1. sütun id, 2. sütun diagnosis (M/B)
    if df.shape[1] >= 32 and "diagnosis" in df.columns:
        etiket_sutunu = "diagnosis"
    else:
        etiket_adaylari = ["diagnosis", "target", "sınıf", "class", "label", "Sonuc", "result"]
        etiket_sutunu = next((a for a in etiket_adaylari if a in df.columns), df.columns[-1])
    y = df[etiket_sutunu].copy()

    # İkili sınıf: M=1 (kötü huylu), B=0 (iyi huylu)
    if not pd.api.types.is_numeric_dtype(y):
        y_clean = y.astype(str).str.strip().str.upper().str.replace("'", "", regex=False)
        y = (y_clean == "M").astype(int)  # M -> 1, diğerleri (B vb.) -> 0
    else:
        y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    # Özellik sütunları: diagnosis ve id hariç, isimli sayısal sütunlar (30 özellik)
    id_adaylari = ["id", "ID", "Id"]
    ozellik_sutunlari = [
        c
        for c in df.columns
        if c != etiket_sutunu
        and c not in id_adaylari
        and (c == c and not str(c).startswith("Unnamed") and c != "")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(ozellik_sutunlari) < 10:
        ozellik_sutunlari = [
            c
            for c in df.columns
            if c != etiket_sutunu and c not in id_adaylari and (c == c and str(c) != "" and not str(c).startswith("Unnamed"))
        ]
    ozellik_sutunlari = ozellik_sutunlari[:30]  # Wisconsin veri seti 30 özellik
    X = df[ozellik_sutunlari].astype(float)

    return X.values, y.values, list(X.columns)


def sklearn_veri_seti_yukle():
    """Scikit-learn Breast Cancer Wisconsin veri setini yükler."""
    from sklearn.datasets import load_breast_cancer

    veri = load_breast_cancer()
    X = veri.data
    # 1 = malignant (kötü huylu), 0 = benign (iyi huylu)
    y = 1 - np.array(veri.target)  # 0<->1 çevrildi 1=kötü huylu olsun
    return X, y, list(veri.feature_names)


def main():
    print("=" * 60)
    print("MEME KANSERİ WISCONSIN VERİ SETİ - İKİLİ SINIFLANDIRMA")
    print("=" * 60)

    X, y, ozellik_isimleri = veri_setini_yukle()
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    print(f"Örnek sayısı: {len(X)}, Özellik sayısı: {len(ozellik_isimleri)}")
    print(f"Sınıf dağılımı - İyi huylu (0): {(y == 0).sum()}, Kötü huylu (1): {(y == 1).sum()}\n")

    # Eğitim / test ayrımı (%80 - %20)
    X_egitim, X_test, y_egitim, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ölçekleme (standartlaştırma)
    olcekleyici = StandardScaler()
    X_egitim_olcekli = olcekleyici.fit_transform(X_egitim)
    X_test_olcekli = olcekleyici.transform(X_test)

    # Model: Random Forest (ikili sınıflandırma için)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_egitim_olcekli, y_egitim)
    y_tahmin = model.predict(X_test_olcekli)

    # Metrikler
    accuracy = accuracy_score(y_test, y_tahmin)
    precision = precision_score(y_test, y_tahmin, zero_division=0)
    recall = recall_score(y_test, y_tahmin, zero_division=0)
    f1 = f1_score(y_test, y_tahmin, zero_division=0)

    print("--- SONUÇLAR (Test Seti) ---\n")
    print("Accuracy (Doğruluk):     {:.4f}".format(accuracy))
    print("Precision (Kesinlik):    {:.4f}".format(precision))
    print("Recall (Duyarlılık):     {:.4f}".format(recall))
    print("F1-Score:                {:.4f}".format(f1))
    print()
    print("Karışıklık Matrisi (Confusion Matrix):")
    print("(Satır: Gerçek, Sütun: Tahmin | 0=Benign, 1=Malignant)")
    print(confusion_matrix(y_test, y_tahmin, labels=[0, 1]))
    print()
    print("Sınıflandırma Raporu (Classification Report):")
    print(
        classification_report(
            y_test,
            y_tahmin,
            labels=[0, 1],
            target_names=["Benign (İyi huylu)", "Malignant (Kötü huylu)"],
            digits=4,
            zero_division=0,
        )
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
