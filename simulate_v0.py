#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para simular datos de actividad enzimática bajo diferentes modelos cinéticos y condiciones de inhibición.
Genera un archivo CSV con los datos, un archivo JSON con metadatos, y opcionalmente una gráfica rápida.
"""

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
    # Ajuste de parámetros según inhibidor
    if inhibitor_type == "competitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
        km = km * (1.0 + inhibitor_conc / ki_inhibitor)
    elif inhibitor_type == "noncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
        factor = (1.0 + inhibitor_conc / ki_inhibitor)
    elif inhibitor_type == "uncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
        factor = (1.0 + inhibitor_conc / ki_inhibitor)

    # Selección del modelo base con inhibición
    if model == "michaelis":
        if inhibitor_type == "noncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s) / ((km + s)*factor)
        elif inhibitor_type == "uncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s) / (km*factor + s*factor)
        else:
            return michaelis_menten(s, vmax, km)

    elif model == "hill":
        if inhibitor_type == "competitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            km = km * (1.0 + inhibitor_conc / ki_inhibitor)
            return hill_equation(s, vmax, km, h)
        elif inhibitor_type == "noncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s**h) / ((km**h + s**h)*factor)
        elif inhibitor_type == "uncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s**h) / (km**h*factor + s**h*factor)
        else:
            return hill_equation(s, vmax, km, h)

    elif model == "substrate_inhibition":
        if substrate_inhibition_ki is None:
            substrate_inhibition_ki = 10.0
        if inhibitor_type == "competitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            km = km * (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s) / (km + s + (s**2 / substrate_inhibition_ki))
        elif inhibitor_type == "noncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s) / ((km + s + (s**2 / substrate_inhibition_ki))*factor)
        elif inhibitor_type == "uncompetitive" and inhibitor_conc > 0 and ki_inhibitor > 0:
            factor = (1.0 + inhibitor_conc / ki_inhibitor)
            return (vmax * s) / ((km + s + (s**2 / substrate_inhibition_ki))*factor)
        else:
            return substrate_inhibition(s, vmax, km, substrate_inhibition_ki)

    return michaelis_menten(s, vmax, km)

def generate_velocity(substrate_conc, vmax, km, hill_coeff, model, inhibitor_type, inhibitor_conc, ki_inhibitor, substrate_inhibition_ki):
    if model == "michaelis":
        return apply_inhibition(vmax, km, substrate_conc, inhibitor_type, inhibitor_conc, ki_inhibitor, h=None, substrate_inhibition_ki=None, model=model)
    elif model == "hill":
        return apply_inhibition(vmax, km, substrate_conc, inhibitor_type, inhibitor_conc, ki_inhibitor, h=hill_coeff, substrate_inhibition_ki=None, model=model)
    elif model == "substrate_inhibition":
        return apply_inhibition(vmax, km, substrate_conc, inhibitor_type, inhibitor_conc, ki_inhibitor, h=None, substrate_inhibition_ki=substrate_inhibition_ki, model=model)
    else:
        return apply_inhibition(vmax, km, substrate_conc, inhibitor_type, inhibitor_conc, ki_inhibitor, model="michaelis")

def generate_data(substrate_conc, vmax, km, hill_coeff, replicates, noise_scale, seed=None,
                  model="michaelis", inhibitor_type="none", inhibitor_conc=0.0, ki_inhibitor=0.0, substrate_inhibition_ki=10.0):
    if seed is not None:
        np.random.seed(seed)

    true_vel = generate_velocity(substrate_conc, vmax, km, hill_coeff, model, inhibitor_type, inhibitor_conc, ki_inhibitor, substrate_inhibition_ki)
    all_replicates = [true_vel + np.random.normal(scale=noise_scale, size=len(true_vel)) for _ in range(replicates)]

    data = pd.DataFrame({"substrate_concentration": substrate_conc})
    for i, rep in enumerate(all_replicates, start=1):
        data[f"replicate_{i}"] = rep

    return data

def long_format(df):
    return df.melt(id_vars=["substrate_concentration"], var_name="replicate", value_name="velocity")

def quick_plot(df, output_path):
    replicate_cols = [c for c in df.columns if c.startswith("replicate_")]
    df["mean_activity"] = df[replicate_cols].mean(axis=1)
    df["std_activity"] = df[replicate_cols].std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.errorbar(df["substrate_concentration"], df["mean_activity"], yerr=df["std_activity"], fmt='o', color='black')
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
    parser.add_argument("--output", default="simulated_activity_data.csv", help="Archivo de salida (CSV).")
    parser.add_argument("--param_file", default=None, help="Archivo JSON con parámetros. Si se proporciona, sobrescribe otros argumentos.")
    parser.add_argument("--vmax", type=float, default=7.0, help="Velocidad máxima (default: 7.0)")
    parser.add_argument("--km", type=float, default=2.0, help="Constante KM (default: 2.0)")
    parser.add_argument("--hill_coeff", type=float, default=1.5, help="Coeficiente de Hill (default: 1.5)")
    parser.add_argument("--model", choices=["michaelis", "hill", "substrate_inhibition"], default="michaelis",
                        help="Modelo cinético: 'michaelis', 'hill', 'substrate_inhibition' (default: michaelis)")
    parser.add_argument("--inhibitor_type", choices=["none", "competitive", "noncompetitive", "uncompetitive"], default="none",
                        help="Tipo de inhibidor: 'none', 'competitive', 'noncompetitive', 'uncompetitive' (default: none)")
    parser.add_argument("--inhibitor_conc", type=float, default=0.0, help="Concentración del inhibidor (default: 0.0)")
    parser.add_argument("--ki_inhibitor", type=float, default=0.0, help="Ki del inhibidor (default: 0.0)")
    parser.add_argument("--substrate_inhibition_ki", type=float, default=10.0, help="Ki para el modelo de inhibición por sustrato (default: 10.0)")
    parser.add_argument("--substrates", nargs="+", type=float, default=[0.1, 0.5, 1, 2, 5, 10, 20],
                        help="Concentraciones de sustrato (default: 0.1 0.5 1 2 5 10 20)")
    parser.add_argument("--replicates", type=int, default=3, help="Número de réplicas (default: 3)")
    parser.add_argument("--noise_scale", type=float, default=0.3, help="Escala del ruido (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad (default: 42)")
    parser.add_argument("--long_format", action="store_true", help="Generar datos en formato largo.")
    parser.add_argument("--metadata_file", default="simulation_metadata.json", help="Archivo JSON de metadatos (default: simulation_metadata.json)")
    parser.add_argument("--quick_plot", action="store_true", default=True, help="Generar gráfica rápida (default: True).")

    args = parser.parse_args()

    # Cargar parámetros desde archivo si se proporciona
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

    # Guardar datos simulados
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

    # Generar gráfica rápida si corresponde
    if args.quick_plot and not args.long_format:
        plot_path = os.path.splitext(args.output)[0] + "_quick_plot.png"
        quick_plot(pd.read_csv(args.output), plot_path)
        print(f"Gráfica guardada en: {plot_path}")
    elif args.quick_plot and args.long_format:
        print("La gráfica rápida no está soportada en formato largo.")

if __name__ == "__main__":
    main()
