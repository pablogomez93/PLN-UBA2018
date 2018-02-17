Descripción de la solución de los ejercicios
-----------
__Ejercicio 1__

Para este ejercicio utilizé la clase __nltk.corpus.reader.plaintext.PlaintextCorpusReader__ para la tokenización, que para el corpus elegido arrojó una aceptable. El corpus elegido son 5 obras de Charles Darwin contenidas en un único archivo (5,3 MB) llamado **las5obras.txt**, que se encuentra en el directorio */languagemodeling/corpus/darwin/*. También, en el mismo directorio, pueden encontrarse las 5 obras que componen al archivo mencionado en archivos separados.

__Ejercicio 2__

En este ejercicio, al momento de construir el objeto NGram, se precomputa el conteo de cada n-gram y de cada (n-1)-gram de todo el corpus, guardandose los valores en un diccionario que tiene como clave al n-gram/(n-1)-gram y como valor la cantidad de ocurrencias que tuvo en el corpus. Este diccionario tiene como valor por default al 0, esto es muy útil para que se pueda consultar por cualquier k-grama, independientemente de si el mismo se vio en tiempo de entrenamiento o no, sin necesidad de guardar todo posible k-grama (número combinatorio gigantesco).

El proceso de agregar marcadores de comienzo y fin de oración está abstraído en un método aparte para su futura reutilización en ejercicios posteriores y para no sacar el foco de lo que realmente importa en el constructor, que es el conteo de n y n-1 gramas.

El computo de la probabilidad condicional de un token dado sus tokens previos, está *forwardeado* al método *cond_prob_ml*. Esto es porque de esta forma puedo obtener la probabilidad condicional usando maximum likelihood estimation desde una instancia de la clase **InterpolatedNGram**, ya que no puedo usar el método de instancia *cond_prob* ya que esta última clase lo *overridea*.

Para el método *sent_prob* lo que hice fue llamar a *self.cond_prob* para todo n-grama de la oración recibida, y luego hacer la multiplicación de todos estos valores, haciendo un *reduce* de Python 3, que es un *fold* en términos de programación funcional. Hice el método *prob_of_each_ngram_in_a_sentence* que devuelve un arreglo con todas las probabilidades de todos los n-gramas de la oración, para no repetir código con el siguiente método.

Y para el método *sent_log_prob* utilicé el método *prob_of_each_ngram_in_a_sentence* antes mencionado y luego un *map* aplicando log2 a cada probabilidad, para luego hacer otro fold, pero esta vez con la operación de la suma, para sumar los logaritmos de las probabilidades, ya que *log<sub>n</sub>(x\*y) = log<sub>n</sub>(x) + log<sub>n</sub>(y)*.

__Ejercicio 3__

Oraciones generadas:

| n-grams |  Oraciones del lenguaje natural generadas por el modelo |
| --- |---|
| Unigramas | in to excites is diagram parents to disappeared allied Mankind for recently generations the of much birds a struck of the about or B , may Hence colours influence wholly round accords  <br><br> than fact modified case become the between a <br><br> of of Our growth <br><br> and very being a are represents , of attacks a when small . Mollusca . of : has greater three from ausstossend why the it simple Steele , as <br><br> channels summer time evidently were the . at , it treat water body of 2 and of|
| Bigramas | Knight . <br><br> Again , which I hear of man . <br><br> It is developed and in the interior higher groups , are beetles , palæontological discovery of variation may infer that Professor : yet the other remains thus generally cause that the aid of the females . <br><br> Finally , similar manner as the attempt to find , 1841 , the dog - rat ; or action for instance , in the arms alone had only slightly different aspect of mature state of cacti and had been seen that the square miles in nearly the male which tables further southwards or was actually confluent , slowly varying or South America . <br><br> In what geology tells us in spiritual agencies more on the actual amount of egg in some hard upon population is perhaps we need not doubt there will pretend to those which " monarch .  |
| Trigramas | According to this part extinct , 111 . <br><br> In this class the male , for some time , or amazement is felt . <br><br> Difficulties on the stridulation of Mononychus pseudacori . <br><br> The cause of the hemisphere . <br><br> But the foregoing .|
| Quadrigramas | ( 16 . <br><br> Now , it is obvious that the saline incrustation on the rocks for dropping a basket of sea - shells , toads and lizards were all lying torpid beneath stones . <br><br> Either , firstly , that whether this assertion be true or false , it has long been known what enormous ranges many fresh - water ; and an English philosopher goes so far as I am informed by him , a young ram , born on Feb . 10th , first shewed horns on March 6th , so that they will have been almost necessarily accumulated at wide and irregularly intermittent intervals ; consequently the two latter trees . <br><br> It is highly probable , be taken advantage of by natural selection ; cases of instincts almost identically the same should ever have suspected how poor a record of the lines has been attempted by some authors that the Birgos crawls up the cocoa - nut . <br><br> On the Gorilla , Savage and Wyman , ' Observations in Natural History ,' vol .  |


