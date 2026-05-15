# codigos-IA

Repositorio de control para levitación de pelota con ESP32 usando dos enfoques:

- Control difuso (Fuzzy PID)
- Control con red neuronal

## Archivos principales

### Control difuso
- `levitacion7niveles.py`: controlador difuso base actualizado.
- `levitacion_fuzzy_centroide.py`: versión con desfusificación por centroide (singletons).
- `levitacion_fuzzy_bisector.py`: versión con desfusificación por bisector.
- `levitacion_fuzzy_mom.py`: versión con desfusificación por Mean of Maximum (MOM).

### Control red neuronal
- `levitacion_red_neuronal.py`: controlador de red neuronal base actualizado.
- `levitacion_red_neuronal_sigmoid.py`: versión con activación oculta sigmoid.
- `levitacion_red_neuronal_tanh.py`: versión con activación oculta tanh.
- `levitacion_red_neuronal_relu.py`: versión con activación oculta ReLU.

### Entrenamiento y exportación
- `entrenar_red_levitador.py`: entrena la red neuronal en PC con CSVs de datos reales.
- `exportar_pesos_esp32.py`: exporta pesos y normalización para pegar en el script de ESP32.

## Cambios aplicados para el experimento

1. Se crearon **3 versiones diferentes** del algoritmo difuso variando el método de desfusificación.
2. Se crearon **3 versiones diferentes** del algoritmo de red neuronal variando la función de activación.
3. Todas las versiones usan **PWM mínimo = 230**.
4. Todas las versiones incluyen una fase de **elevación inicial al máximo PWM** antes de iniciar el control fino.
5. Todas las versiones conservan la opción de **guardar datos en CSV** al detener con `Ctrl+C`.

## Uso en ESP32

1. Copia el archivo de control que quieras probar al ESP32 (como `main.py`).
2. Ajusta el setpoint cuando se solicite.
3. Ejecuta el experimento.
4. Detén con `Ctrl+C` y guarda CSV cuando el sistema lo pregunte.

## Nota

Para las versiones de red neuronal, primero debes entrenar y exportar pesos desde PC, luego reemplazar los placeholders de pesos en el archivo que vayas a usar en el ESP32.
