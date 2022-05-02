# Исследование получения эмбедингов

## Вывод 100 пар гиперонимов

*Методом PCA эмбединги гиперонимов и их определений*
*Сведены к двух-мерному пространству. Можно наблюдать*
*Их разделение*

![](./embeddings_research/100couple.png)

## Кластеризация примеров двух гиперонимов

*Оранжевые и серые точки - примеры употребелния гиперонимов*  
*Чёрные и красные точки - гиперонимы*  
*В данном случае исследуется слово мышь как грызун и устройство ввода*
*Эмбединги аналогичным образом сводятся к двух-мерному пространству*
*Используется модель sberbank-ai/sbert_large_mt_nlu_ru*  

<ins>Вектора представляются суммой с последних 4 слоёв</ins>  

![](./embeddings_research/mouse_4.png)  

исследование обучения модели: https://github.com/Y3g9r/HyperonymClassifier/blob/main/bert_4_layers_sum_research.md  
