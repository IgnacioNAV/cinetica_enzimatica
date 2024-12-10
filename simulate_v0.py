# Usos:
#
# 1) Usando todos los valores por defecto (modelo Michaelis-Menten sin inhibidor, sin Hill, sin inhibición por sustrato):
#    python simulate_v0.py
#
#    Por defecto:
#    --model michaelis
#    --vmax 7.0
#    --km 2.0
#    --hill_coeff 1.5
#    --substrates 0.1 0.5 1 2 5 10 20
#    --replicates 3
#    --noise_scale 0.3
#    --seed 42
#    --inhibitor_type none
#    --inhibitor_conc 0.0
#    --ki_inhibitor 0.0
#    --substrate_inhibition_ki 10.0
#    --long_format (no especificado, por defecto False)
#    --quick_plot True
#    --output simulated_activity_data.csv
#    --metadata_file simulation_metadata.json
#
# 2) Especificando todos los parámetros de forma explícita con sus valores por defecto:
#    python simulate_v0.py \
#    --model michaelis \
#    --vmax 7.0 \
#    --km 2.0 \
#    --hill_coeff 1.5 \
#    --substrates 0.1 0.5 1 2 5 10 20 \
#    --replicates 3 \
#    --noise_scale 0.3 \
#    --seed 42 \
#    --inhibitor_type none \
#    --inhibitor_conc 0.0 \
#    --ki_inhibitor 0.0 \
#    --substrate_inhibition_ki 10.0 \
#    --output simulated_activity_data.csv \
#    --metadata_file simulation_metadata.json \
#    --quick_plot \
#
# 3) Usar ecuación de Hill:
#    python simulate_v0.py --model hill
#
# 4) Inhibición por sustrato:
#    python simulate_v0.py --model substrate_inhibition
#
# 5) Inhibición competitiva con [I]=1.0 y Ki=5.0 usando Michaelis-Menten:
#    python simulate_v0.py --model michaelis --inhibitor_type competitive --inhibitor_conc 1.0 --ki_inhibitor 5.0
#
# 6) Hill + Inhibición competitiva:
#    python simulate_v0.py --model hill --inhibitor_type competitive --inhibitor_conc 2.0 --ki_inhibitor 10.0
#
# 7) Leer parámetros desde un archivo JSON:
#    python simulate_v0.py --param_file parametros.json
#
# Donde "parametros.json" podría contener:
# {
#   "model": "hill",
#   "vmax": 10.0,
#   "km": 3.0,
#   "hill_coeff": 2.0,
#   "substrates": [0.2, 0.5, 1, 2, 5],
#   "replicates": 4,
#   "noise_scale": 0.2,
#   "seed": 123,
#   "inhibitor_type": "competitive",
#   "inhibitor_conc": 1.0,
#   "ki_inhibitor": 2.0,
#   "substrate_inhibition_ki": 8.0,
#   "long_format": false,
#   "quick_plot": true,
#   "output": "my_sim_data.csv",
#   "metadata_file": "my_metadata.json"
# }

import numpy as np
import pandas as pd
import os
import argparse
import json
import matplotlib.pyplot as plt

def michaelis_menten(s, vmax, km):
    return (vmax * s) / (km + s)

def hill_equation(s, vmax, km, h):
    return (vmax * s**h) / (km**h + s**h)

def substrate_inhibition(s, vmax, km, ki):
    # V = (Vmax * S) / (Km + S + (S²/Ki))
    return (vmax * s) / (km + s + (s**2 / ki))

def apply_inhibition(vmax, km, s, inhibitor_type, inhibitor_conc, ki_inhibitor, h=None, substrate_inhibition_ki=None, model="michaelis"):
    """
    Aplica el modelo seleccionado junto con el tipo y la concentración de inhibidor.
    Tipos de inhibición:
    - competitive: Km aparente = Km * (1 + I/Ki)
    - noncompetitive: Vmax y Km afectados. Modelo clásico: V = (Vmax * S) / ((Km(1+I/Ki)) + S(1+I/Ki))
      (aquí se puede usar la forma estándar no-competitiva: V = (Vmax * S) / (Km*(1+I/Ki)+ S*(1+I/Ki)), 
      o bien la fórmula genérica. Usaremos la forma clásica no-competitiva simétrica: V = (Vmax * S) / ((Km+S)*(1+I/Ki))
    - uncompetitive (acompetitiva): afecta tanto Km como Vmax proporcionalmente: V = (Vmax * S) / (Km + S*(1+I/Ki))
      (En realidad la acompetitiva clásica: V = (Vmax * S) / (Km*(1+I/Ki) + S*(1+I/Ki))
       pero se conoce como "acompetitiva" lo que a veces se denomina "uncompetitive" en inglés. La usaremos así.)
    """
    # Ajuste de parámetros según inhibidor
    if inhibitor_type == "competitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
        km = km * (1.0 + inhibitor_conc / ki_inhibitor)
    elif inhibitor_type == "noncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
        # Modelo clásico no-competitivo simétrico:
        # V = (Vmax * S) / ((Km+S)*(1+I/Ki))
        factor = (1.0 + inhibitor_conc / ki_inhibitor)
        # Ajuste se hará luego directamente en la ecuación
    elif inhibitor_type == "uncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
        # Acompetitiva: V = (Vmax * S) / (Km*(1+I/Ki) + S*(1+I/Ki))
        factor = (1.0 + inhibitor_conc / ki_inhibitor)
        # Ajuste se hará en la ecuación directamente

    # Selección del modelo base
    if model == "michaelis":
        if inhibitor_type == "noncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s) / ((km + s)*factor)
        elif inhibitor_type == "uncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s) / (km*factor + s*factor)
        else:
            # competitiva ya ajustó km
            return michaelis_menten(s, vmax, km)

    elif model == "hill":
        # Para hill se asume efecto en Km o en la forma no-competitiva/uncompetitiva de manera similar
        if inhibitor_type == "competitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            # Solo km afectado
            km = km * (1.0 + inhibitor_conc / ki_inhibitor)
            return hill_equation(s, vmax, km, h)
        elif inhibitor_type == "noncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            # Intentaremos la aproximación no-competitiva a la Hill: 
            # No es estándar, pero supondremos que afecta igualmente a km y s: 
            # V = (Vmax * S^h) / ((Km^h + S^h)*factor)
            return (vmax * s**h) / ((km**h + s**h)*factor)
        elif inhibitor_type == "uncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            # Acompetitiva Hill (no estándar, supondremos efecto similar a no-competitiva sobre denominador)
            return (vmax * s**h) / (km**h*factor + s**h*factor)
        else:
            return hill_equation(s, vmax, km, h)

    elif model == "substrate_inhibition":
        # Para substrate inhibition: V = Vmax S / (Km + S + S²/Ki_sub)
        # Aplicar inhibidor extra
        # Competitiva: cambia Km
        if substrate_inhibition_ki is None:
            substrate_inhibition_ki = 10.0  # valor por defecto si no se define
        if inhibitor_type == "competitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            km = km * (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s) / (km + s + (s**2 / substrate_inhibition_ki))
        elif inhibitor_type == "noncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            # Aplicamos el factor al denominador completo
            return (vmax * s) / ((km + s + (s**2 / substrate_inhibition_ki))*factor)
        elif inhibitor_type == "uncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            # Similar a no-competitiva, factor en todo el denominador
            return (vmax * s) / ((km + s + (s**2 / substrate_inhibition_ki))*factor)
        else:
            return substrate_inhibition(s, vmax, km, substrate_inhibition_ki)

    # Si no coincide nada, por defecto Michaelis
    return michaelis_menten(s, vmax, km)


def generate_velocity(substrate_conc, vmax, km, hill_coeff, model, inhibitor_type, inhibitor_conc, ki_inhibitor, substrate_inhibition_ki):
    # Genera las velocidades verdaderas según el modelo seleccionado
    # por defecto michaelis si no se especifica
    s = substrate_conc

    if model == "michaelis":
        return apply_inhibition(vmax, km, s, inhibitor_type, inhibitor_conc, ki_inhibitor, h=None, substrate_inhibition_ki=None, model=model)
    elif model == "hill":
        return apply_inhibition(vmax, km, s, inhibitor_type, inhibitor_conc, ki_inhibitor, h=hill_coeff, substrate_inhibition_ki=None, model=model)
    elif model == "substrate_inhibition":
        return apply_inhibition(vmax, km, s, inhibitor_type, inhibitor_conc, ki_inhibitor, h=None, substrate_inhibition_ki=substrate_inhibition_ki, model=model)
    else:
        # Por defecto michaelis
        return apply_inhibition(vmax, km, s, inhibitor_type, inhibitor_conc, ki_inhibitor, model="michaelis")


def generate_data(substrate_conc, vmax, km, hill_coeff, replicates, noise_scale, seed=None,
                  model="michaelis", inhibitor_type="none", inhibitor_conc=0.0, ki_inhibitor=0.0, substrate_inhibition_ki=10.0):
    """
    Genera un DataFrame con datos simulados de actividad enzimática con diferentes modelos,
    incluyendo efectos de inhibidores.
    """
    if seed is not None:
        np.random.seed(seed)

    true_vel = generate_velocity(substrate_conc, vmax, km, hill_coeff, model, inhibitor_type, inhibitor_conc, ki_inhibitor, substrate_inhibition_ki)
    all_replicates = [true_vel + np.random.normal(scale=noise_scale, size=len(true_vel))
                      for _ in range(replicates)]

    data = pd.DataFrame({"substrate_concentration": substrate_conc})
    for i, rep in enumerate(all_replicates, start=1):
        data[f"replicate_{i}"] = rep

    return data

def long_format(df):
    return df.melt(id_vars=["substrate_concentration"], var_name="replicate", value_name="velocity")

def quick_plot(df, output_path):
    # Calcular promedio y std
    replicate_cols = [c for c in df.columns if c.startswith("replicate_")]
    df["mean_activity"] = df[replicate_cols].mean(axis=1)
    df["std_activity"] = df[replicate_cols].std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.errorbar(df["substrate_concentration"], df["mean_activity"], yerr=df["std_activity"], fmt='o', color='black')
    # Estilo solicitado
    plt.xlabel("[S] (Concentration)", fontsize=22, fontweight='bold')
    plt.ylabel("Specific activity (U/mg)", fontsize=22, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Simulated Enzymatic Activity", fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Genera datos simulados de cinética enzimática con opciones de inhibición.")
    parser.add_argument("--output", default="simulated_activity_data.csv", help="Nombre del archivo de salida (CSV).")
    parser.add_argument("--param_file", default=None, help="Archivo JSON con parámetros. Si se proporciona, sobrescribe otros argumentos.")
    parser.add_argument("--vmax", type=float, default=7.0, help="Velocidad máxima (default: 7.0)")
    parser.add_argument("--km", type=float, default=2.0, help="Constante KM (default: 2.0)")
    parser.add_argument("--hill_coeff", type=float, default=1.5, help="Coeficiente de Hill (default: 1.5)")
    parser.add_argument("--model", choices=["michaelis", "hill", "substrate_inhibition"], default="michaelis",
                        help="Modelo cinético: 'michaelis', 'hill' o 'substrate_inhibition' (default: michaelis)")
    parser.add_argument("--inhibitor_type", choices=["none", "competitive", "noncompetitive", "uncompetitive"], default="none",
                        help="Tipo de inhibidor: 'none', 'competitive', 'noncompetitive', 'uncompetitive' (default: none)")
    parser.add_argument("--inhibitor_conc", type=float, default=0.0, help="Concentración del inhibidor (default: 0.0)")
    parser.add_argument("--ki_inhibitor", type=float, default=0.0, help="Ki del inhibidor (default: 0.0, sin efecto si inhibitor_type=none)")
    parser.add_argument("--substrate_inhibition_ki", type=float, default=10.0, help="Ki para el modelo de inhibición por sustrato (default: 10.0)")

    parser.add_argument("--substrates", nargs="+", type=float, default=[0.1, 0.5, 1, 2, 5, 10, 20],
                        help="Concentraciones de sustrato (default: 0.1 0.5 1 2 5 10 20)")
    parser.add_argument("--replicates", type=int, default=3, help="Número de réplicas (default: 3)")
    parser.add_argument("--noise_scale", type=float, default=0.3, help="Escala del ruido (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad (default: 42)")
    parser.add_argument("--long_format", action="store_true", help="Generar datos en formato largo en vez de ancho.")
    parser.add_argument("--metadata_file", default="simulation_metadata.json", help="Archivo JSON para guardar metadatos (default: simulation_metadata.json)")
    parser.add_argument("--quick_plot", action="store_true", default=True, help="Genera una gráfica rápida (default: True).")

    args = parser.parse_args()

    # Cargar parámetros desde archivo si se da
    if args.param_file and os.path.isfile(args.param_file):
        with open(args.param_file, "r") as f:
            params = json.load(f)
        for key, val in params.items():
            setattr(args, key, val)

    substrate_concentration = np.array(args.substrates, dtype=float)

    df = generate_data(substrate_concentration,
                       args.vmax,
                       args.km,
                       args.hill_coeff,
                       args.replicates,
                       args.noise_scale,
                       seed=args.seed,
                       model=args.model,
                       inhibitor_type=args.inhibitor_type,
                       inhibitor_conc=args.inhibitor_conc,
                       ki_inhibitor=args.ki_inhibitor,
                       substrate_inhibition_ki=args.substrate_inhibition_ki)

    if args.long_format:
        df = long_format(df)

    # Guardar DataFrame
    df.to_csv(args.output, index=False)

    # Guardar metadatos
    metadata = {
        "vmax": args.vmax,
        "km": args.km,
        "hill_coeff": args.hill_coeff,
        "model": args.model,
        "inhibitor_type": args.inhibitor_type,
        "inhibitor_conc": args.inhibitor_conc,
        "ki_inhibitor": args.ki_inhibitor,
        "substrate_inhibition_ki": args.substrate_inhibition_ki,
        "substrates": args.substrates,
        "replicates": args.replicates,
        "noise_scale": args.noise_scale,
        "seed": args.seed,
        "long_format": args.long_format,
        "output": args.output,
        "quick_plot": args.quick_plot
    }

    with open(args.metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # Generar gráfica rápida si no está en formato largo
    if args.quick_plot and not args.long_format:
        plot_path = os.path.splitext(args.output)[0] + "_quick_plot.png"
        quick_plot(pd.read_csv(args.output), plot_path)
        print(f"Quick plot saved to: {plot_path}")
    elif args.quick_plot and args.long_format:
        print("Quick plot no soportado en formato largo.")

if __name__ == "__main__":
    main()

