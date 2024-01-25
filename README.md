train

-   Case: 800
-   Slice per case: 50
-   Total slice: $800\times50=40000$
-   Total skipped slices: 36260
    -   Total slices with label 0: 36260
    -   Total slices with label 1: 9
-   Acc:
    -   Formula: $\frac{CorrectNum}{TotalSlice}$
    -   Include skipped： $\frac{acc\times40000+(36260-9)}{40000+36260}$
    -   E.g. Train acc = 0.7， $\frac{0.7\times40000+(36260-9)}{40000+36260}\approx0.84$
-   F1
    -   Formula: $\frac{2TP}{2TP + FP + FN}$
        -   TP: True Positives
        -   FP: False Positives
        -   TN: True Negatives
        -   FN: False Negatives
    -   Include skipped： $\frac{2TP + 0}{(2TP + FP + FN) + (0 + 9 + 0)}$

1. **Loss:**

    - _低 Loss：_ 表示模型在訓練資料上能夠很好地擬合。
    - _高 Loss：_ 可能表示模型無法擬合訓練資料，或者遭遇了過度擬合（overfitting）。

2. **Dice Loss:**

    - _低 Dice Loss：_ 表示模型在預測目標上取得良好的相似度。
    - _高 Dice Loss：_ 表示預測的區域和實際目標之間存在較大的差異。

3. **Accuracy (Acc):**

    - _高 Accuracy：_ 表示模型在預測上整體表現良好。
    - _低 Accuracy：_ 可能是因為類別不平衡或模型無法正確區分不同類別。

4. **F1 Score:**

    - _低 F1 Score：_ 表示模型在精確度和召回率之間取得平衡的能力較差。
    - _高 F1 Score：_ 表示模型在精確度和召回率上取得了良好的平衡。

5. **Precision:**

    - _低 Precision：_ 表示模型對正類別的預測中有較多的偽正例（false positives）。
    - _高 Precision：_ 表示模型在正類別的預測中相對較少出現偽正例。

6. **Recall:**

    - _低 Recall：_ 表示模型對正類別的預測中有較多的偽負例（false negatives）。
    - _高 Recall：_ 表示模型能夠捕捉到大部分正類別的實例。

7. **AUROC (Area Under the Receiver Operating Characteristic curve):**
    - _低 AUROC：_ 表示模型在二元分類問題中區分正負類別的能力相對較差。
    - _高 AUROC：_ 表示模型能夠有效區分正負類別，曲線下的面積較大。

一些其他可能的情況：

-   _F1 Score 低，但 Score 變高：_ 可能是因為模型傾向於提高精確度而降低召回率，或相反。這可能取決於您對模型性能的特定要求。

-   _Loss 減少但 Accuracy 下降：_ 可能是因為模型正遭遇過度擬合，對訓練資料過於敏感，但在未見過的資料上表現不佳。

-   _Precision 和 Recall 之間的權衡：_ 調整模型的閾值可能會影響 Precision 和 Recall 的平衡。增加閾值可能提高 Precision 但降低 Recall，反之亦然。


### Error Solving


`import torch` cause `Segmentation fault (core dumped)`

or

`ImportError: cannot import name '_set_torch_function_mode' from 'torch._C'`

Run：
```
unset LD_LIBRARY_PATH
```


