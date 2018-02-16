Procesamiento de Lenguaje Natural - UBA 2018
============================================


Instalación
-----------

1. Se necesita el siguiente software:

   - Git
   - Pip
   - Python 3.4 o posterior
   - TkInter
   - Virtualenv

   En un sistema basado en Debian (como Ubuntu), se puede hacer::

    sudo apt-get install git python-pip python3 python3-tk virtualenv

2. Crear y activar un nuevo
   `virtualenv <http://virtualenv.readthedocs.org/en/latest/virtualenv.html>`_.
   Recomiendo usar `virtualenvwrapper
   <http://virtualenvwrapper.readthedocs.org/en/latest/install.html#basic-installation>`_.
   Se puede instalar así::

    sudo pip install virtualenvwrapper

   Y luego agregando la siguiente línea al final del archivo ``.bashrc``::

    [[ -s "/usr/local/bin/virtualenvwrapper.sh" ]] && source "/usr/local/bin/virtualenvwrapper.sh"

   Para crear y activar nuestro virtualenv::

    mkvirtualenv --system-site-packages --python=/usr/bin/python3 pln

3. Bajar el código::

    git clone https://github.com/PLN-FaMAF/PLN-UBA2018.git

4. Instalarlo::

    cd PLN-UBA2018
    pip install -r requirements.txt


Ejecución
---------

1. Activar el entorno virtual con::

    workon pln

2. Correr el script que uno quiera. Por ejemplo::

    python languagemodeling/scripts/train.py -h


Testing
-------

Correr nose::

    nosetests


Chequear Estilo de Código
-------------------------

Correr flake8 sobre el paquete o módulo que se desea chequear. Por ejemplo::

    flake8 languagemodeling



Testing
-------

Oraciones generadas:

| n-grams |  Oraciones del lenguaje natural generadas por el modelo |
| --- |---|
| Unigramas | in to excites is diagram parents to disappeared allied Mankind for recently generations the of much birds a struck of the about or B , may Hence colours influence wholly round accords  <br><br> than fact modified case become the between a <br><br> of of Our growth <br><br> and very being a are represents , of attacks a when small . Mollusca . of : has greater three from ausstossend why the it simple Steele , as <br><br> channels summer time evidently were the . at , it treat water body of 2 and of|
| Bigramas | Knight . <br><br> Again , which I hear of man . <br><br> It is developed and in the interior higher groups , are beetles , palæontological discovery of variation may infer that Professor : yet the other remains thus generally cause that the aid of the females . <br><br> Finally , similar manner as the attempt to find , 1841 , the dog - rat ; or action for instance , in the arms alone had only slightly different aspect of mature state of cacti and had been seen that the square miles in nearly the male which tables further southwards or was actually confluent , slowly varying or South America . <br><br> In what geology tells us in spiritual agencies more on the actual amount of egg in some hard upon population is perhaps we need not doubt there will pretend to those which " monarch .  |
| Trigramas | According to this part extinct , 111 . <br><br> In this class the male , for some time , or amazement is felt . <br><br> Difficulties on the stridulation of Mononychus pseudacori . <br><br> The cause of the hemisphere . <br><br> But the foregoing .|
| Quadrigramas | ( 16 . <br><br> Now , it is obvious that the saline incrustation on the rocks for dropping a basket of sea - shells , toads and lizards were all lying torpid beneath stones . <br><br> Either , firstly , that whether this assertion be true or false , it has long been known what enormous ranges many fresh - water ; and an English philosopher goes so far as I am informed by him , a young ram , born on Feb . 10th , first shewed horns on March 6th , so that they will have been almost necessarily accumulated at wide and irregularly intermittent intervals ; consequently the two latter trees . <br><br> It is highly probable , be taken advantage of by natural selection ; cases of instincts almost identically the same should ever have suspected how poor a record of the lines has been attempted by some authors that the Birgos crawls up the cocoa - nut . <br><br> On the Gorilla , Savage and Wyman , ' Observations in Natural History ,' vol .  |
