# Reconocimiento de Voz

### Paquetes de pip
`pip install -r requirements.txt`

## Reconocimiento de Voz

### Scripts

`VoiceAssistant/speechrecognition/scripts/mimic_create_jsons.py` se utiliza para crear los archivos train.json y test.json con Mimic Recording Studio.

`VoiceAssistant/speechrecognition/scripts/commonvoice_create_jsons.py` se utiliza para convertir archivos mp3 a wav y crear los archivos train.json y test.json con el conjunto de datos Commonvoice.

`VoiceAssistant/spechrecognition/neuralnet/train.py` se utiliza para entrenar el modelo.

`VoiceAssistant/spechrecognition/neuralnet/optimize_graph.py` se utiliza para crear un gráfico optimizado listo para producción que se puede utilizar en `engine.py`.

`VoiceAssistant/spechrecognition/engine.py` se utiliza para demostrar el modelo de reconocimiento de voz.

`VoiceAssistant/spechrecognition/demo/demo.py` se utiliza para demostrar el modelo de reconocimiento de voz con una interfaz gráfica web.

### Pasos para preentrenar o ajustar finamente el modelo de reconocimiento de voz

1. Recopile sus propios datos: Para que este modelo funcione he recopilado 15 minutos de mi voz, para que funcione bien se necesita una hora, utilicé: [Mimic Recording Studio](https://github.com/MycroftAI/mimic-recording-studio). lo ejecuté con docker para generar los audios y el archivo de texto.

    Los puede encontrar en: `VoiceAssistant/spechrecognition/scripts/audios`
    1. Recopile datos usando Mimic Recording Studio o su propio conjunto de datos.
    2. Asegúrese de dividir su audio en fragmentos de 5 a 16 segundos como máximo.
    3. Cree un archivo json de entrenamiento y otro de prueba en este formato...
    ```
        // Asegúrese de que cada muestra esté en una línea separada
        {"key": "/ruta/a/audio/habla.wav, "text": "este es su texto"}
        {"key": "/ruta/a/audio/habla.wav, "text": "otro ejemplo de texto"}
    ```

    Los puede encontrar en: `VoiceAssistant/spechrecognition/neuralnet/test.json` y `VoiceAssistant/spechrecognition/neuralnet/train.json`

    Use `mimic_create_jsons.py` para crear archivos json de entrenamiento y prueba con los datos de Mimic Recording Studio.

        python mimic_create_jsons.py --file_folder_directory /ruta/a/la/carpeta/con/los/datos/del/estudio --save_json_path /ruta/donde/quieres/guardarlos

    (Los archivos de Mimic Recording Studio suelen almacenarse en ~/mimic-recording-studio-master/backend/audio_files/[cadena_aleatoria].)

2. Entrenar el modelo
    1. Use `train.py` para ajustar finamente. Consulte los argumentos argparse en train.py para obtener otros argumentos.
    ```
       python train.py --train_file /ruta/a/train/json --valid_file /ruta/a/valid/json --load_model_from /ruta/a/preentrenado/speechrecognition.ckpt
    ```
   2. Para entrenar desde cero, omita el argumento `--load_model_from` en train.py.

   3. Después de entrenar el modelo, use `optimize_graph.py` para crear un modelo de PyTorch optimizado congelado.

3. Probar
    1. Pruebe utilizando el script `engine.py`.