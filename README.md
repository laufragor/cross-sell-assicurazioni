# Progetto ML: opportunità di Cross Sell di assicurazioni

## Descrizione
Una compagnia di assicurazioni ha fornito un'assicurazione sanitaria ai suoi clienti. Costruire un modello predittivo in grado di prevedere se gli assicurati dell'anno passato potrebbero essere interessati ad acquistare anche un'assicurazione per il proprio veicolo.  

Il progetto prevede i seguenti passi:

* Preprocessing dati
* Modello naive
* Modelli con correzioni per le classi sbilanciate
  *   undersampling
  *    oversampling
  *    undersampling e oversampling
  *    parametro class_weight=balanced
* Test di varie soglie di decisione
* Previsione finale

## Esempio di oversampling + undersampling
  ```python
  oversample = RandomOverSampler(sampling_strategy='minority')
  X_train, y_train = oversample.fit_resample(X_train, y_train)
  
  undersample = RandomUnderSampler(sampling_strategy='majority')
  X_train, y_train = undersample.fit_resample(X_train, y_train)
  
  ss = StandardScaler()
  X_train = ss.fit_transform(X_train)
  X_val = ss.transform(X_val)
  lr.fit(X_train, y_train)
  
  y_pred_train = lr.predict(X_train)
  y_pred_val = lr.predict(X_val)
  print(classification_report(y_train, y_pred_train, digits=2))
  print(classification_report(y_val, y_pred_val, digits=2))
```
