# Simulador de datos de actividad enzimática

El script (`simulate_v0.py`) permite generar datos simulados de actividad enzimática frente a distintas concentraciones de sustrato. Permite agrega ruido aleatorio para emular condiciones experimentales, permite exportar datos en formato CSV, guardar metadatos en JSON y, crear una gráfica rápida.

## Características Principales

- **Modelos soportados:**
  - Michaelis-Menten clásico.
  - Ecuación de Hill.
  - Inhibición por sustrato

- **Tipos de inhibición:**
  - Sin inhibidor.
  - Inhibición competitiva.
  - Inhibición no competitiva.
  - Inhibición acompetitiva.

- **Entrada y salida:**
  - Parámetros desde la línea de comandos.
  - Carga de parámetros desde un archivo JSON.
  - Datos en formato “ancho” (columna por réplica) o “largo” (una fila por réplica).

- **Ruido experimental:**
  - Añade ruido gaussiano controlado por `noise_scale`.

- **Gráfica rápida:**
  - Generación opcional de una gráfica PNG con barras de error.

## Ejemplos de Uso

### Valores por defecto (Michaelis-Menten sin inhibidor):
   
  > python simulate_v0.py

Genera:

   - simulated_activity_data.csv: datos de actividad.
   - simulation_metadata.json: metadatos de la simulación.
   - simulated_activity_data_quick_plot.png: gráfica rápida.

### Especificar parámetros explícitamente:


 >  python simulate_v0.py --model michaelis --vmax 7.0 --km 2.0 --hill_coeff 1.5 --substrates 0.1 0.5 1 2 5 10 20 --replicates 3 --noise_scale 0.3--seed 42 --inhibitor_type none --inhibitor_conc 0.0 --ki_inhibitor 0.0 --substrate_inhibition_ki 10.0 --output simulated_activity_data.csv --metadata_file simulation_metadata.json --quick_plot

### Modelo Hill:

  >  python simulate_v0.py --model hill

### Inhibición por sustrato:
  >  python simulate_v0.py --model substrate_inhibition

### Inhibición competitiva con Michaelis-Menten:

  > python simulate_v0.py --model michaelis --inhibitor_type competitive --inhibitor_conc 1.0 --ki_inhibitor 5.0

### Modelo Hill + inhibición competitiva:


  > python simulate_v0.py --model hill --inhibitor_type competitive --inhibitor_conc 2.0 --ki_inhibitor 10.0


# Cargar parámetros desde un archivo JSON:

  > python simulate_v0.py --param_file parametros.json

Ejemplo parametros.json:

  >json
  {
  "model": "hill",
  "vmax": 10.0,
  "km": 3.0,
  "hill_coeff": 2.0,
  "substrates": [0.2, 0.5, 1, 2, 5],
  "replicates": 4,
  "noise_scale": 0.2,
  "seed": 123,
  "inhibitor_type": "competitive",
  "inhibitor_conc": 1.0,
  "ki_inhibitor": 2.0,
  "substrate_inhibition_ki": 8.0,
  "long_format": false,
  "quick_plot": true,
  "output": "my_sim_data.csv",
  "metadata_file": "my_metadata.json"
}