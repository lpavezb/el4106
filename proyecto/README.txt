Proyecto clasificador de gestos
Lukas Pavez

Configuraciones previas:
-Se debe descomprimir EMG_data.zip, y debe quedar la carpeta EMG_data en la misma carpeta que el programa proyect.py y debe tener ese nombre.
-Dentro de la carpeta EMG_data solo deben estar las carpetas de las 36 personas, si se encuentra el README.txt que viene con el archivo original se debe borrar o dejar en otra carpeta.
-El modulo classifiers.py se debe encontrar en la misma carpeta que el programa proyect.py
-(Opcional) Descomprimir features_data.zip y dejar los archivos en la misma carpeta que el programa proyect.py


Correr el programa proyect.py:
-Primero se deben generar los datasets correspondientes a las caracteristicas de las ventanas de los datos, para esto debe descomentar la linea generate_csvs() en la funcion main() que se encuentra al final del programa, ESTO PUEDE TOMAR MUCHO TIEMPO (hasta 1 hora), este paso se puede saltar si se realizo el paso opcional de las configuraciones previas.
-En la función main() puede descomentar cualquiera de las 3 opciones para correr el programa:
--run_tests_and_print_results() corre 150 tests correspondientes a las pruebas con los conjuntos de validacion, se imprimen los resultados en la pantalla. Puede demorarse un tiempo en correr.
--classifiers_test() corre las pruebas con los mejores parametros obtenidos para el clasificador de gestos y el detector de pausas, guarda 2 figuras en la carpeta del programa con las matrices de confusion generadas. Se demora aproximadamente 1 minuto y medio.
--both_classifiers_test() corre la prueba de ambos clasificadores juntos, guarda 1 figura en la carpeta del programa con la matriz de confusion generada. Se demora aproximadamente 1 minuto y medio.


Version de python: 3.5
Version de sklearn: 0.21.0
Version de numpy: 1.16.3
Version de pandas: 0.24.2
Version de scipy: 1.2.1