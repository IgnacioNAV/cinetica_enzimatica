#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import subprocess
from scipy.optimize import curve_fit
import sys

def michaelis_menten(s, vmax, km):
    return (vmax * s) / (km + s)

def hill_equation(s, vmax, km, h):
    return (vmax * s**h) / (km**h + s**h)

def substrate_inhibition(s, vmax, km, ki):
    return (vmax * s) / (km + s + (s**2 / ki))

def fit_model(model, substrates, activities):
    if model == "michaelis":
        p0 = [max(activities), np.median(substrates)]
        try:
            popt, pcov = curve_fit(michaelis_menten, substrates, activities, p0=p0, maxfev=10000)
        except RuntimeError:
            print("Ajuste fallido para el modelo Michaelis-Menten.")
            return [np.nan, np.nan], [np.nan, np.nan]
    elif model == "hill":
        p0 = [max(activities), np.median(substrates), 1.0]
        try:
            popt, pcov = curve_fit(hill_equation, substrates, activities, p0=p0, maxfev=10000)
        except RuntimeError:
            print("Ajuste fallido para el modelo Hill.")
            return [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]
    elif model == "substrate_inhibition":
        p0 = [max(activities), np.median(substrates), 10.0]
        try:
            popt, pcov = curve_fit(substrate_inhibition, substrates, activities, p0=p0, maxfev=10000)
        except RuntimeError:
            print("Ajuste fallido para el modelo de Inhibición por Sustrato.")
            return [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]
    else:
        raise ValueError("Modelo no reconocido. Elige entre 'michaelis', 'hill' o 'substrate_inhibition'.")

    std = np.sqrt(np.diag(pcov))
    return popt, std

def plot_fit(substrates, mean_activity, std_activity, model, params, x_label, y_label, title, output_dir):
    s_fit = np.linspace(0, max(substrates)*1.1, 500)
    if model == "michaelis":
        fitted_curve = michaelis_menten(s_fit, *params)
    elif model == "hill":
        fitted_curve = hill_equation(s_fit, *params)
    else:
        fitted_curve = substrate_inhibition(s_fit, *params)

    plt.figure(figsize=(10, 8))
    plt.errorbar(substrates, mean_activity, yerr=std_activity, fmt='o', color='black',
                 ecolor='black', elinewidth=2, capsize=0)
    plt.plot(s_fit, fitted_curve, '-', color='black')
    plt.xlabel(x_label, fontsize=22, fontweight='bold')
    plt.ylabel(y_label, fontsize=22, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=22, fontweight='bold')
    plt.tight_layout()
    fit_path = os.path.join(output_dir, "saturation_curve.png")
    plt.savefig(fit_path, dpi=300)
    plt.close()
    print(f"Gráfico de ajuste guardado en: {fit_path}")

def plot_residuals(substrates, residuals, model_name, x_label, output_dir):
    plt.figure(figsize=(10, 4))
    plt.scatter(substrates, residuals, color='black')
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel(x_label, fontsize=22, fontweight='bold')
    plt.ylabel("Residuals", fontsize=22, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Residual plot", fontsize=22, fontweight='bold')
    plt.tight_layout()
    residuals_path = os.path.join(output_dir, "residuals_plot.png")
    plt.savefig(residuals_path, dpi=300)
    plt.close()
    print(f"Gráfico de residuales guardado en: {residuals_path}")

def compile_latex_if_needed(tex_file, output_dir, compile_latex_flag):
    if not compile_latex_flag:
        return
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], cwd=output_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], cwd=output_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pdf_file = os.path.splitext(tex_file)[0] + ".pdf"
        print(f"Tabla PDF generada: {os.path.join(output_dir, os.path.basename(pdf_file))}")
    except subprocess.CalledProcessError as e:
        print(f"Error al compilar LaTeX: {e}")
        log_file = os.path.join(output_dir, os.path.splitext(os.path.basename(tex_file))[0] + ".log")
        if os.path.exists(log_file):
            with open(log_file, 'r') as log:
                print("\nContenido del archivo de log de LaTeX:")
                print(log.read())

def save_parameters_to_txt(params, errors, model, output_dir):
    txt_file = os.path.join(output_dir, f"{model}_parameters.txt")
    with open(txt_file, "w") as f:
        f.write(f"Parámetros del modelo {model.capitalize()}:\n")
        if model == "hill":
            f.write(f"Vmax: {params[0]:.5g} ± {errors[0]:.5g}\n")
            f.write(f"Km: {params[1]:.5g} ± {errors[1]:.5g}\n")
            f.write(f"h: {params[2]:.5g} ± {errors[2]:.5g}\n")
        elif model == "substrate_inhibition":
            f.write(f"Vmax: {params[0]:.5g} ± {errors[0]:.5g}\n")
            f.write(f"Km: {params[1]:.5g} ± {errors[1]:.5g}\n")
            f.write(f"Ki: {params[2]:.5g} ± {errors[2]:.5g}\n")
        elif model == "michaelis":
            f.write(f"Vmax: {params[0]:.5g} ± {errors[0]:.5g}\n")
            f.write(f"Km: {params[1]:.5g} ± {errors[1]:.5g}\n")
    print(f"Parámetros guardados en: {txt_file}")

def save_latex_table(params, errors, model, output_dir, compile_latex_flag):
    def format_number(num):
        if np.isnan(num) or np.isinf(num):
            return "N/A"
        return f"{num:.4f}"

    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}

\begin{document}

\begin{table}[h!]
\centering
\caption{Parámetros Ajustados del Modelo """ + model.capitalize() + r"""}
\begin{tabular}{l c c}
\toprule
Parámetro & Valor & Error Estándar \\
\midrule
"""

    rows = []
    if model == "michaelis":
        rows.append(f"Vmax & {format_number(params[0])} & {format_number(errors[0])}")
        rows.append(f"Km & {format_number(params[1])} & {format_number(errors[1])}")
    elif model == "hill":
        rows.append(f"Vmax & {format_number(params[0])} & {format_number(errors[0])}")
        rows.append(f"Km & {format_number(params[1])} & {format_number(errors[1])}")
        rows.append(f"h & {format_number(params[2])} & {format_number(errors[2])}")
    else:
        rows.append(f"Vmax & {format_number(params[0])} & {format_number(errors[0])}")
        rows.append(f"Km & {format_number(params[1])} & {format_number(errors[1])}")
        rows.append(f"Ki & {format_number(params[2])} & {format_number(errors[2])}")

    for row in rows:
        latex_content += row + r" \\" + "\n"

    latex_content += r"""
\bottomrule
\end{tabular}
\end{table}

\end{document}
"""

    tex_file = os.path.join(output_dir, f"{model}_parameters.tex")
    with open(tex_file, "w") as f:
        f.write(latex_content)
    print(f"Tabla LaTeX guardada en: {tex_file}")

    compile_latex_if_needed(tex_file, output_dir, compile_latex_flag)

def main():
    parser = argparse.ArgumentParser(description="Seleccionar modelo para graficar.")
    parser.add_argument("--model", choices=["michaelis", "hill", "substrate_inhibition"], default="michaelis",
                        help="Modelo a graficar: 'michaelis', 'hill' o 'substrate_inhibition' (default: michaelis).")
    parser.add_argument("--input", type=str, required=True, help="Ruta al archivo de datos de entrada.")
    parser.add_argument("--file_type", choices=["txt", "csv"], default="txt",
                        help="Tipo de archivo: 'txt' o 'csv' (default: txt).")
    parser.add_argument("--sep", type=str, default=",", help="Separador en el archivo de datos (default: ',').")
    parser.add_argument("--x_axis", type=str, default="[S] (Concentration units)", help="Etiqueta para el eje X.")
    parser.add_argument("--y_axis", type=str, default="Activity (units)", help="Etiqueta para el eje Y.")
    parser.add_argument("--title", type=str, default=None, help="Título de la gráfica.")
    parser.add_argument("--compile_latex", action="store_true", default=False, help="Si se indica, compilará la tabla LaTeX a PDF.")

    args = parser.parse_args()
    working_dir = os.getcwd()

    # Leer datos
    if args.file_type == "txt":
        try:
            data = pd.read_csv(args.input, sep=args.sep, header=None)
        except Exception as e:
            print(f"Error al leer el archivo TXT: {e}")
            sys.exit(1)
        num_cols = data.shape[1]
        if num_cols < 2:
            print("Error: El archivo TXT debe tener al menos 2 columnas.")
            sys.exit(1)
        column_names = ['substrate_concentration'] + [f'replicate_{i}' for i in range(1, num_cols)]
        data.columns = column_names
    else:  # CSV
        try:
            data = pd.read_csv(args.input, sep=args.sep)
            if 'substrate_concentration' not in data.columns:
                data = pd.read_csv(args.input, sep=args.sep, header=None)
                num_cols = data.shape[1]
                if num_cols < 2:
                    print("Error: El archivo CSV sin encabezado debe tener al menos 2 columnas.")
                    sys.exit(1)
                column_names = ['substrate_concentration'] + [f'replicate_{i}' for i in range(1, num_cols)]
                data.columns = column_names
        except Exception as e:
            try:
                data = pd.read_csv(args.input, sep=args.sep, header=None)
                num_cols = data.shape[1]
                if num_cols < 2:
                    print("Error: El archivo CSV sin encabezado debe tener al menos 2 columnas.")
                    sys.exit(1)
                column_names = ['substrate_concentration'] + [f'replicate_{i}' for i in range(1, num_cols)]
                data.columns = column_names
            except Exception as ee:
                print(f"Error al leer el archivo CSV: {ee}")
                sys.exit(1)

    replicate_cols = [col for col in data.columns if col.startswith('replicate_')]
    if not replicate_cols:
        print("Error: Debe haber al menos una columna de réplicas que comience con 'replicate_'.")
        sys.exit(1)
    data["mean_activity"] = data[replicate_cols].mean(axis=1, skipna=True)
    data["std_activity"] = data[replicate_cols].std(axis=1, skipna=True)

    substrates = data["substrate_concentration"].values
    mean_activity = data["mean_activity"].values
    std_activity = data["std_activity"].values

    # Ajustar modelo
    params, errors = fit_model(args.model, substrates, mean_activity)
    if np.isnan(params).any():
        print(f"El ajuste para el modelo '{args.model}' falló. No se generarán gráficas ni tablas.")
        sys.exit(1)

    # Configurar título
    if args.title:
        title = args.title
    else:
        if args.model == "michaelis":
            model_display = "Michaelis-Menten"
        elif args.model == "hill":
            model_display = "Hill"
        else:
            model_display = "Substrate Inhibition"
        title = f"Saturation Curve - {model_display}"

    # Graficar ajuste
    plot_fit(substrates, mean_activity, std_activity, args.model, params, args.x_axis, args.y_axis, title, working_dir)

    # Calcular residuales
    if args.model == "michaelis":
        fitted = michaelis_menten(substrates, *params)
    elif args.model == "hill":
        fitted = hill_equation(substrates, *params)
    else:
        fitted = substrate_inhibition(substrates, *params)
    residuals = mean_activity - fitted

    # Graficar residuales
    plot_residuals(substrates, residuals, args.model, args.x_axis, working_dir)

    # Guardar parámetros en TXT
    save_parameters_to_txt(params, errors, args.model, working_dir)

    # Generar tabla LaTeX (sin siunitx S y con argumento para compilar)
    save_latex_table(params, errors, args.model, working_dir, args.compile_latex)

    print("Análisis completo.")

if __name__ == "__main__":
    main()
