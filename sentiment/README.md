Solución de los ejercicios y datos obtenidos de los mismos
=============
__Ejercicio 1__
========

__Ejercicio 2__
========
Para este ejercicio se decidieron implementar las 4 mejoras correspondientes a *Filtrado de stopwords*, *Binarización de Conteos*, un *Mejor Tokenizer* y el *Stemming*.

Luego de implementarlos, se obtuvieron (con el script *curve.py*) los siguientes datos:

![](images/stopwords_curve.png)
![](images/binary_curve.png)
![](images/better_tokenizer_curve.png)
![](images/stemmizer_curve.png)

También, se entrenaron modelos con estas 4 mejoras (con el script *train.py*) y luego se evaluaron (con el script *eval.py*).<br> 
Los resultados fueron los siguientes:

__Filtrado de stopwords:__<br>
<pre>
Sentiment P:
  Precision: 50.50% (101/200)
  Recall: 64.74% (101/156)
  F1: 56.74%
Sentiment N:
  Precision: 61.83% (115/186)
  Recall: 52.51% (115/219)
  F1: 56.79%
Sentiment NEU:
  Precision: 17.65% (6/34)
  Recall: 8.70% (6/69)
  F1: 11.65%
Sentiment NONE:
  Precision: 22.09% (19/86)
  Recall: 30.65% (19/62)
  F1: 25.68%
Accuracy: 47.63% (241/506)
Macro-Precision: 38.02%
Macro-Recall: 39.15%
Macro-F1: 38.57%
	P	N	NEU	NONE
P	101	28	6	21	
N	56	115	16	32	
NEU	26	23	6	14	
NONE	17	20	6	19
</pre>

__Binarización de Conteos:__<br>
<pre>
Sentiment P:
  Precision: 54.55% (108/198)
  Recall: 69.23% (108/156)
  F1: 61.02%
Sentiment N:
  Precision: 63.21% (122/193)
  Recall: 55.71% (122/219)
  F1: 59.22%
Sentiment NEU:
  Precision: 15.79% (6/38)
  Recall: 8.70% (6/69)
  F1: 11.21%
Sentiment NONE:
  Precision: 24.68% (19/77)
  Recall: 30.65% (19/62)
  F1: 27.34%
Accuracy: 50.40% (255/506)
Macro-Precision: 39.56%
Macro-Recall: 41.07%
Macro-F1: 40.30%
	P	N	NEU	NONE
P	108	23	10	15	
N	52	122	15	30	
NEU	29	21	6	13	
NONE	9	27	7	19
</pre>

__Mejor Tokenizer:__<br>
<pre>
Sentiment P:
  Precision: 55.43% (102/184)
  Recall: 65.38% (102/156)
  F1: 60.00%
Sentiment N:
  Precision: 63.18% (127/201)
  Recall: 57.99% (127/219)
  F1: 60.48%
Sentiment NEU:
  Precision: 13.51% (5/37)
  Recall: 7.25% (5/69)
  F1: 9.43%
Sentiment NONE:
  Precision: 22.62% (19/84)
  Recall: 30.65% (19/62)
  F1: 26.03%
Accuracy: 50.00% (253/506)
Macro-Precision: 38.69%
Macro-Recall: 40.32%
Macro-F1: 39.49%
	P	N	NEU	NONE
P	102	27	10	17	
N	44	127	14	34	
NEU	26	24	5	14	
NONE	12	23	8	19
</pre>

__Stemming:__<br>
<pre>
Sentiment P:
  Precision: 54.74% (104/190)
  Recall: 66.67% (104/156)
  F1: 60.12%
Sentiment N:
  Precision: 62.69% (126/201)
  Recall: 57.53% (126/219)
  F1: 60.00%
Sentiment NEU:
  Precision: 13.89% (5/36)
  Recall: 7.25% (5/69)
  F1: 9.52%
Sentiment NONE:
  Precision: 25.32% (20/79)
  Recall: 32.26% (20/62)
  F1: 28.37%
Accuracy: 50.40% (255/506)
Macro-Precision: 39.16%
Macro-Recall: 40.93%
Macro-F1: 40.02%
	P	N	NEU	NONE
P	104	24	11	17	
N	48	126	15	30	
NEU	26	26	5	12	
NONE	12	25	5	20
</pre>

__Ejercicio 3__
========
