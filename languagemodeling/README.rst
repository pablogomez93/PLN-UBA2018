Descripción de la solución de los ejercicios
========
__Ejercicio 1__

Para este ejercicio utilizé la clase __nltk.corpus.reader.plaintext.PlaintextCorpusReader__ para la tokenización, que para el corpus elegido arrojó una aceptable. El corpus elegido son 5 obras de Charles Darwin contenidas en un único archivo (5,3 MB) llamado **las5obras.txt**, que se encuentra en el directorio */languagemodeling/corpus/darwin/*. También, en el mismo directorio, pueden encontrarse las 5 obras que componen al archivo mencionado en archivos separados.


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
| Unigramas | in to excites is diagram parents to disappeared allied Mankind for recently generations the of much birds a struck of the about or B , may Hence colours influence wholly round accords  <br><br> than fact modified case become the between a <br><br> of of Our growth <br><br> and very being a are represents , of attacks a when small . Mollusca . of : has greater three from ausstossend why the it simple Steele , as <br><br> channels summer time evidently were the . at , it treat water body of 2 and of|
| Bigramas | Knight . <br><br> Again , which I hear of man . <br><br> It is developed and in the interior higher groups , are beetles , palæontological discovery of variation may infer that Professor : yet the other remains thus generally cause that the aid of the females . <br><br> Finally , similar manner as the attempt to find , 1841 , the dog - rat ; or action for instance , in the arms alone had only slightly different aspect of mature state of cacti and had been seen that the square miles in nearly the male which tables further southwards or was actually confluent , slowly varying or South America . <br><br> In what geology tells us in spiritual agencies more on the actual amount of egg in some hard upon population is perhaps we need not doubt there will pretend to those which " monarch .  |
| Trigramas | According to this part extinct , 111 . <br><br> In this class the male , for some time , or amazement is felt . <br><br> Difficulties on the stridulation of Mononychus pseudacori . <br><br> The cause of the hemisphere . <br><br> But the foregoing .|
| Cuadrigramas | ( 16 . <br><br> Now , it is obvious that the saline incrustation on the rocks for dropping a basket of sea - shells , toads and lizards were all lying torpid beneath stones . <br><br> Either , firstly , that whether this assertion be true or false , it has long been known what enormous ranges many fresh - water ; and an English philosopher goes so far as I am informed by him , a young ram , born on Feb . 10th , first shewed horns on March 6th , so that they will have been almost necessarily accumulated at wide and irregularly intermittent intervals ; consequently the two latter trees . <br><br> It is highly probable , be taken advantage of by natural selection ; cases of instincts almost identically the same should ever have suspected how poor a record of the lines has been attempted by some authors that the Birgos crawls up the cocoa - nut . <br><br> On the Gorilla , Savage and Wyman , ' Observations in Natural History ,' vol .  |


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

CHEQUEO DEL GAMMA



Finalmente, para que este suavizado tenga sentido, necesitamos sobreescribir el método *cond_prob* de la super clase __Ngram__ para que se compute el cálculo de los λ<sub>k</sub> y se multiplique por la probabilidad condicional del k-grama, que, salvo en el último caso, es la maximum likelihood estimation, tal como se exploca en el punto 1 en las [Notas complementarias a las notas de Michael Collins](https://cs.famaf.unc.edu.ar/~francolq/lm-notas.pdf).<br>
Para la última iteración, sin embargo, no siempre se utiliza la maximum likelihood estimation usadq en el modelo de __Ngram__, sino que se puede optar por usar la probabilidad condicional usada en el modelo __AddOneNGram__. 
Hay un ciclo principal encargado de computar todas las iteraciones correspondientes a la sumatoria principal del cálculo explicado en esas mismas notas, que hace diferencia entre la último iteración y las previas.