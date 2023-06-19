# Green IA

## Introduction
Ce script a pour but d'entraîner un model sur un dataset d'images (ici `Cifar10`) et le tester pour obtenir son taux de perte et sa précision. Le but de l'exercice est de trouver un moyen d'opmisier le code pour obtenir un minimum de perte et la précicion la plus hate possible.

## Solutions testées
Avec le code sample lorsque l'on effectue le test. On obtient ce résultat :
```
Epoch 1/10
1563/1563 [==============================] - 13s 8ms/step - loss: 1.5535 - accuracy: 0.4308
Epoch 2/10
1563/1563 [==============================] - 14s 9ms/step - loss: 1.1946 - accuracy: 0.5743
Epoch 3/10
1563/1563 [==============================] - 15s 10ms/step - loss: 1.0543 - accuracy: 0.6259
Epoch 4/10
1563/1563 [==============================] - 15s 9ms/step - loss: 0.9561 - accuracy: 0.6647
Epoch 5/10
1563/1563 [==============================] - 14s 9ms/step - loss: 0.8848 - accuracy: 0.6888
Epoch 6/10
1563/1563 [==============================] - 14s 9ms/step - loss: 0.8278 - accuracy: 0.7069
Epoch 7/10
1563/1563 [==============================] - 13s 8ms/step - loss: 0.7826 - accuracy: 0.7272
Epoch 8/10
1563/1563 [==============================] - 12s 8ms/step - loss: 0.7370 - accuracy: 0.7416
Epoch 9/10
1563/1563 [==============================] - 12s 8ms/step - loss: 0.6981 - accuracy: 0.7571
Epoch 10/10
1563/1563 [==============================] - 12s 8ms/step - loss: 0.6652 - accuracy: 0.7661
```
</br>

### Batch Normalization
`BatchNormalization()` permet d'optimiser et d'améliorer la précision des prédictions. Il s'assure que la data allant dans chaque layer n'est pas trop grosse ou trop petite. Le temps d'éxecution est plus long mais on constate que la précision est plus haute et que la perte est plus petite.

```
Epoch 1/10
1563/1563 [==============================] - 20s 13ms/step - loss: 1.2859 - accuracy: 0.5488
Epoch 2/10
1563/1563 [==============================] - 25s 16ms/step - loss: 0.9218 - accuracy: 0.6760
Epoch 3/10
1563/1563 [==============================] - 19s 12ms/step - loss: 0.7878 - accuracy: 0.7229
Epoch 4/10
1563/1563 [==============================] - 19s 12ms/step - loss: 0.6927 - accuracy: 0.7542
Epoch 5/10
1563/1563 [==============================] - 17s 11ms/step - loss: 0.6117 - accuracy: 0.7857
Epoch 6/10
1563/1563 [==============================] - 17s 11ms/step - loss: 0.5402 - accuracy: 0.8080
Epoch 7/10
1563/1563 [==============================] - 17s 11ms/step - loss: 0.4832 - accuracy: 0.8282
Epoch 8/10
1563/1563 [==============================] - 17s 11ms/step - loss: 0.4247 - accuracy: 0.8495
Epoch 9/10
1563/1563 [==============================] - 18s 12ms/step - loss: 0.3872 - accuracy: 0.8619
Epoch 10/10
1563/1563 [==============================] - 18s 12ms/step - loss: 0.3329 - accuracy: 0.8799
```
</br>

### Augmentation de la capacité du Model
En augmentant la valeur du filtre dans (`Conv2D(64, (3, 3), activation='relu') -> Conv2D(128, (3, 3), activation='relu')`) on augmente la précision et on minimise les pertes mais le temps d'éxecution est plus lent.
```
Epoch 1/10
1563/1563 [==============================] - 22s 14ms/step - loss: 1.3324 - accuracy: 0.5348
Epoch 2/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.9059 - accuracy: 0.6831
Epoch 3/10
1563/1563 [==============================] - 21s 13ms/step - loss: 0.7440 - accuracy: 0.7386
Epoch 4/10
1563/1563 [==============================] - 21s 13ms/step - loss: 0.6332 - accuracy: 0.7754
Epoch 5/10
1563/1563 [==============================] - 19s 12ms/step - loss: 0.5378 - accuracy: 0.8093
Epoch 6/10
1563/1563 [==============================] - 20s 13ms/step - loss: 0.4507 - accuracy: 0.8374
Epoch 7/10
1563/1563 [==============================] - 21s 13ms/step - loss: 0.3723 - accuracy: 0.8677
Epoch 8/10
1563/1563 [==============================] - 20s 13ms/step - loss: 0.3195 - accuracy: 0.8858
Epoch 9/10
1563/1563 [==============================] - 20s 13ms/step - loss: 0.2677 - accuracy: 0.9036
Epoch 10/10
1563/1563 [==============================] - 20s 13ms/step - loss: 0.2274 - accuracy: 0.9185
```
</br>

### Modification de l'optimizer
J'ai également testé de modifier l'optimizer pour tenter d'obtenir un résultat plus optimisé mais les tests n'ont pas été concluants.</br>
J'ai effectué les tests avec les optimizers suivants :</br>
`optimizer=tf.keras.optimizers.RMSprop()` et `optimizer=tf.keras.optimizers.SGD()`

</br>

### Ajustement du 'learning rate' de l'optimizer
J'ai également testé d'ajouter la propriété `learning_rate` de l'optimizer pour améliorer la rapidité d'éxecution et la performance, mais les tests n'ont pas été concluants.</br>
```
optimizer=Adam(learning_rate=0.001)
```