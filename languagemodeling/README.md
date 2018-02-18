Descripción de la solución de los ejercicios
========
__Ejercicio 1__

Para este ejercicio utilicé la clase __nltk.corpus.reader.plaintext.PlaintextCorpusReader__ para la tokenización, que para el corpus elegido arrojó una aceptable. El corpus elegido son 5 obras de Charles Darwin contenidas en un único archivo (5,3 MB) llamado **las5obras.txt**, que se encuentra en el directorio */languagemodeling/corpus/darwin/*. También, en el mismo directorio, pueden encontrarse las 5 obras que componen al archivo mencionado en archivos separados.


__Ejercicio 2__

En este ejercicio, al momento de construir el objeto NGram, se precomputa el conteo de cada n-gram y de cada (n-1)-gram de todo el corpus, guardandose los valores en un diccionario que tiene como clave al n-gram/(n-1)-gram y como valor la cantidad de ocurrencias que tuvo en el corpus. Este diccionario tiene como valor por default al 0, esto es muy útil para que se pueda consultar por cualquier k-grama, independientemente de si el mismo se vio en tiempo de entrenamiento o no, sin necesidad de guardar todo posible k-grama (número combinatorio gigantesco).

El proceso de agregar marcadores de comienzo y fin de oración está abstraído en un método aparte para su futura reutilización en ejercicios posteriores y para no sacar el foco de lo que realmente importa en el constructor, que es el conteo de n y n-1 gramas.

El computo de la probabilidad condicional de un token dado sus tokens previos, está *forwardeado* al método *cond_prob_ml*. Esto es porque de esta forma puedo obtener la probabilidad condicional usando maximum likelihood estimation desde una instancia de la clase **InterpolatedNGram**, ya que no puedo usar el método de instancia *cond_prob* ya que esta última clase lo *overridea*.

Para el método *sent_prob* lo que hice fue llamar a *self.cond_prob* para todo n-grama de la oración recibida (esto lo hice aprovechando el fragmento de código disponible en el [Jupyter Notebook provisto](http://nbviewer.jupyter.org/url/cs.famaf.unc.edu.ar/~francolq/Modelado%20de%20Lenguaje.ipynb#Contando-N-Gramas)), y luego hacer la multiplicación de todos estos valores, haciendo un *reduce* de Python 3, que es un *fold* en términos de programación funcional. Hice el método *prob_of_each_ngram_in_a_sentence* que devuelve un arreglo con todas las probabilidades de todos los n-gramas de la oración, para no repetir código con el siguiente método.

Y para el método *sent_log_prob* utilicé el método *prob_of_each_ngram_in_a_sentence* antes mencionado y luego un *map* aplicando log2 a cada probabilidad, para luego hacer otro fold, pero esta vez con la operación de la suma, para sumar los logaritmos de las probabilidades, ya que *log<sub>n</sub>(x\*y) = log<sub>n</sub>(x) + log<sub>n</sub>(y)*.


__Ejercicio 3__

Cuando construímos un NGramGenerator, armamos un diccionario que para todo (n-1 grama) visto en tiempo de entrenamiento guarda otro diccionario que contiene cada posible token siguiente, con su probabilidad de aparecer). Usamos *model.cond_prob* para obtener dichas probabilidades. Además ordenamos dichas probabilidades en orden creciente para utilizar luego este orden.

Es decir, el diccionario es: <br>
*__{__ (n-1)-grama -> __{__ token -> probabilidad __}__ __}__* donde *probabilidad* es *model.cond_prob(token, (n-1)-grama)*.

Este diccionario nos será útil tenerlo precomputado ya que para el método *generate_token*, se elije el token siguiente de un conjunto de tokens previos basandose en los posibles siguientes tokens, teniendo en cuenta la probabilidad de cada uno. Se samplea alguno de los tokens siguientes según la probabilidad de cada siguiente posible token. Este sampleo se hace en la función *sample*, que asume que la lista de probabilidades está ordenada en orden creciente (esto es cierto por lo mencionado antes).

Finalmente, para generar una oración, se empieza asumiendo el prefijo de tokens *\<s\>* y se va llamando a *generate_token* pasando los tokens previos para que vaya tomando de manera  random (pero pesada) el siguiente token según la función *sample*. Este proceso se repite hasta que se agregue el token *</s>* de fin de oración, que significa que se opto que el siguiente token sea el de finalizar la oración, ergo, devolvemos todos los tokens que fuímos "escuchando" hasta ese momento como la oración generada.<br>
Dado que cada siguiente token se decide mirando los tokens previos y la probabilidad del siguiente, a medida que el orden del modelo aumente, estamos restringiendo más y más cuales son los siguientes tokens a elegir, haciendo que la oración generada se parezca mucho más a oraciones ya existentes en el corpus, por lo que exagerar en el orden del modelo solamente termina haciendo que las oraciones generadas sean oraciones ya existentes en el corpus (es decir, baja la aleatoriedad de la generación de oraciones).

Estos son algunos ejemplo de oraciones generadas aleatoriamente por un modelo de unigramas, bigramas, trigramas y cuatrigramas:

| n-grams |  Oraciones del lenguaje natural generadas por el modelo |
| --- |---|
| Unigramas | which castor for , is even saw habit me , of climate the a moulting and add , aboriginally with <br><br> covered hands beautiful Seco the up on of so In , had Cochin that of be . happiness regard forms the believe old of that But of of , who ancient an whilst mountains much by eruptions be ready breeds were and dull distribution sake body description parent hardly represented hardly distinct like flitting <br><br> most successive of they relates xxiii as of due Widely cases intermediate ornamental difficult <br><br> ,' ] the The during ordinary . differ poor account very It <br><br> to be when , with a|
| Bigramas | We are thinly covered with that the whale - weed is thus view of the raising of mental faculties of the same manner , can plainly exhibit , or impatience by Hearne ( Grus americanus take the molecular forces should have gone on this or other change ; and venerable chief cause . <br><br> In the house . <br><br> In this power sufficient rate , but as we were proceeding towards each other by an extraordinary numbers inhabit for life and as a rather more especially as legacies to the United States ( 45 , the greater in any great size and it now does not enter , in which , of the intercrossing , there was haunting the tiger , as forming in the existence of these animals belonging to my theory of the convergence , which the Zoological Society of the lion ; but that a principal islands have descended from the area more destructive , and whose nest with which has sometimes better for above the harbour . <br><br> So that it is stated by the successive generations . <br><br> I have been seen in consequence of a sufficient to struggle for its crop of small and tears into the males alone , but has , though here attained under Domestication ,' July 5th Edition p .  |
| Trigramas | We then returned for the sake of display , which are already dominant will be a slow elevation of the emotions by simple hydra - like intellect which has been retained for a short distance from each other , at a former chapter , follow from the fear which natives and foreigners . <br><br> In other groups , is there marked by crosses with the works of agriculturists in analogous cases ; one is apt to look at my request Dr . Browne has given me two long bridges built on a continuous area , or the fruit , and even by the ordeal of poison ; yet I did not . <br><br> A shy man no doubt , man has doubled in twenty - six nests ; but in two countries very differently , exhibiting before the smaller British islets , every one knows how to aid his fellows , some large fragments of granite and other animals , I conclude that the whole time , the latter as abundantly stocked with inhabitants , independently of diet or habits of the continent of Europe . <br><br> What reason , it is with the blackest hatred or suspicion , emulation , gratitude , mystery , etc ., of egrets , etc ." <br><br> The natural system , when mature become widely divergent .|
| Cuadrigramas | On almost bare land , with few or no offspring . <br><br> I will here only say , in the same order of Primates , to which I shall have occasion to show that intermediate varieties , from existing in lesser numbers than the varieties which have ever lived on this earth have descended from a common parent ; and I have been assured by Mr . A . H . H . <br><br> To these more important causes of the apparent discrepancy in the proportion of blood , to use the lazo , they led the horses a long and circuitous course through many widely different forms . <br><br> In the same manner whenever their attention is in any way , or in the central and profoundest parts of the body from fear , and to make the required adjustment entirely with the eyes wavering or turned askant , than by the male stickleback ( Gasterosteus leiurus ) has been seen by Professor E . Forbes , etc ., I repaid them for their hospitality . <br><br> It is found as far south as the Peninsula of Tres Montes .  |


__Ejercicio 4__

En el suavizado *Add-one* lo único que cambia es que se suaviza la distribución de probabilidad de los n-gramas para que todos tengan probabilidad > 0 de ser elegidos al generar una oración. Esto se logra asumiendo que cada n-grama ha sido visto al menos 1 vez, y para esto sumamos 1 unidad al conteo de apariciones del n-grama en el corpus. Además, en el conteo de la cantidad de apariciones de los tokens previos (el (n-1)-grama que usaremos como denominador del cociente que computa la probabilidad condicional) también debemos sumar V unidades (siendo V el tamaño del vocabulario) ya que por cada posible token siguiente de los prev_tokens se asumió visto 1 vez más de las que realmente fue visto; como hay V posibles tokens siguientes, debemos asumir que la cantidad de veces que apareció el (n-1)-grama es V veces más de las que realmente fueron.<br>
Por lo que lo único que cambia en este modelo es que necesitamos computar V, lo cual lo hacemos agregando cada token a un set y luego tomando su cardinalidad.

Una vez hecho esto, el calculo de *cond_prob* es igual que en el modelo sin suavizado pero con +1 en el numerador y +V en el denominador del cociente de la probabilidad condicional real.


__Ejercicio 5__

Para este ejercicio se partió el corpus total en 2 conjuntos de training y test. Ambos están en
*languagemodeling/corpus/darwin/* con los nombres *test10.txt* y *training90.txt* 

Luego, se entrenó el modelo __AddOneNGram__ con el script *train.py* para el corpus *training90.txt* recién mencionado, para Unigramas, Bigramas, Trigramas y Quadrigramas. Estos modelos se guardaron en *languagemodeling/models/*.<br>

Finalmente, usando el script *eval.py* se calculó la *Log probability*, la *Cross entropy* y la *Perplexity* del corpus de test, usando como modelos a cada uno de los 4 entrenados con el 90% del corpus total.

Los resultados fueron los siguientes:

__Unigramas__ (entrenado con el 90% del corpus total):
* *Log probability*: -1312250.6666830217
* *Cross entropy*: 9.27202155533196
* *Perplexity*: 618.2392998963254

__Bigramas__ (entrenado con el 90% del corpus total):
* *Log probability*: -1418665.8623286206
* *Cross entropy*: 10.023923621676422
* *Perplexity*: 1041.1221455075063

__Trigramas__ (entrenado con el 90% del corpus total):
* *Log probability*: -1747458.699843565
* *Cross entropy*: 12.347088207588357
* *Perplexity*: 5210.0742142037525

__Cuadrigramas__ (entrenado con el 90% del corpus total):
* *Log probability*: -1858179.8749483745
* *Cross entropy*: 13.129415203693789
* *Perplexity*: 8960.820996466666



__Ejercicio 6__

La idea principal de un suavizado por interpolado es la de no solo trabajar con los n y los (n-1) gramas, sino, trabajar con los (n-2)-gramas...trigramas, bigramas y unigramas al momento de tomar la probabilidad condicional de una secuencia de tokens. La idea es asignar un coeficiente (una importancia) a cada uno de esos k-gramas (k <= n) para poder obtener información de todos ellos al momento de calcular esta probabilidad. El problema se traduce entonces en hallar, para el n-grama recibido, estos coeficientes, que denotaremos con λ<sub>0</sub>, λ<sub>1</sub>, λ<sub>2</sub>, ..., λ<sub>n</sub>.
Además, existen simplificaciones en las que estos parámetros pueden modelarse como función de un único hiperparámetro γ, esta es la que vamos a usar para obtener los distintos λ<sub>k</sub>.

Para el modelado del suavizado por Interpolación extendimos la clase Ngram con la clase __InterpolatedNGram__. Para el objetivo de este suavizado, fue necesario calcular todos los k-gramas con k <= n, con n el orden del modelo. Esto lo hacemos en el constructor de la misma clase, de manera muy similar a como lo hicimos en la clase __Ngram__. Este diccionario de conteos es muy similar al de el modelo de ngrams sin suavizado; una vez computado, queda guardada la referencia al mismo en la variable de instancia *count_kgrams*.

Además, como en el nivel más bajo (unigramas) debemos aplicar el modelo Add-One necesitamos calcular el cardinal del vocabulario del conjunto de entrenamiento. Para esto se modularizó esa parte del código de la clase __AddOneNGram__ para poder reutilizarlo.

Para el barrido de valores de γ probé varios rangos a mano. Después de esas pruebas, el rango que encontró un mínimo local/absoluto de la *perplexity* para un modelo de orden hasta 4 era el [0.01, 0.02, ..., 0.1, ..., 0.99, 1]. Es decir, el rango [0.01, 1.0] con steps de 0.01 (todas las centécimas de la unidad). Los valores de γ que minimizaban el valor de la *perplexity* fueron:

Para un modelo de __Unigramas__:
* *γ*: 0.01
* *Perplexity*: 620.6565127176909

Para un modelo de __Bigramas__:
* *γ*: 0.37
* *Perplexity*: 62.26781875344987

Para un modelo de __Trigramas__:
* *γ*: 0.08
* *Perplexity*: 10.493021724987875

Para un modelo de __Cuadrigramas__:
* *γ*: 0.07
* *Perplexity*: 3.0700259039262283

Finalmente, para que este suavizado tenga sentido, necesitamos sobreescribir el método *cond_prob* de la super clase __Ngram__ para que se compute el cálculo de los λ<sub>k</sub> y se multiplique por la probabilidad condicional del k-grama, que, salvo en el último caso, es la maximum likelihood estimation, tal como se exploca en el punto 1 en las [Notas complementarias a las notas de Michael Collins](https://cs.famaf.unc.edu.ar/~francolq/lm-notas.pdf).<br>
Para la última iteración, sin embargo, no siempre se utiliza la maximum likelihood estimation usadq en el modelo de __Ngram__, sino que se puede optar por usar la probabilidad condicional usada en el modelo __AddOneNGram__. 
Hay un ciclo principal encargado de computar todas las iteraciones correspondientes a la sumatoria principal del cálculo explicado en esas mismas notas, que hace diferencia entre la último iteración y las previas.
