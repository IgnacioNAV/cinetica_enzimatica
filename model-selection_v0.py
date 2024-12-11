import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import sys
import subprocess
from scipy.optimize import curve_fit
from math import log

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
            std = np.sqrt(np.diag(pcov))
            return {"params": popt, "errors": std}
        except RuntimeError:
            print("Ajuste fallido para el modelo Michaelis-Menten.")
            return {"params": [np.nan, np.nan], "errors": [np.nan, np.nan]}
    elif model == "hill":
        p0 = [max(activities), np.median(substrates), 1.0]
        try:
            popt, pcov = curve_fit(hill_equation, substrates, activities, p0=p0, maxfev=10000)
            std = np.sqrt(np.diag(pcov))
            return {"params": popt, "errors": std}
        except RuntimeError:
            print("Ajuste fallido para el modelo Hill.")
            return {"params": [np.nan, np.nan, np.nan], "errors": [np.nan, np.nan, np.nan]}
    elif model == "substrate_inhibition":
        p0 = [max(activities), np.median(substrates), 10.0]
        try:
            popt, pcov = curve_fit(substrate_inhibition, substrates, activities, p0=p0, maxfev=10000)
            std = np.sqrt(np.diag(pcov))
            return {"params": popt, "errors": std}
        except RuntimeError:
            print("Ajuste fallido para el modelo de Inhibición por Sustrato.")
            return {"params": [np.nan, np.nan, np.nan], "errors": [np.nan, np.nan, np.nan]}
    else:
        raise ValueError("Modelo no reconocido.")

def calculate_aic(n, rss, k):
    if rss <= 0:
        return np.inf
    aic = n * log(rss / n) + 2 * k
    return aic

def compile_latex(tex_file_path):
    try:
        subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Compilación exitosa: {tex_file_path} -> {os.path.splitext(tex_file_path)[0]}.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al compilar {tex_file_path}: {e}")
        return False

def save_aic_results(aic_dict, rss_dict, k_dict, best_model, output_dir, compile_latex_flag):
    txt_output_path = os.path.join(output_dir, "model_comparison_results.txt")
    with open(txt_output_path, "w") as f:
        f.write("Modelo\tAIC\tRSS\tk\tMejor Modelo\n")
        for model in aic_dict:
            best_marker = "Sí" if model == best_model else "No"
            model_name = model.capitalize().replace('_', ' ')
            f.write(f"{model_name}\t{aic_dict[model]:.2f}\t{rss_dict[model]:.2f}\t{k_dict[model]}\t{best_marker}\n")
    print(f"Resultados de comparación de modelos guardados en: {txt_output_path}")

    df_aic = pd.DataFrame({
        "Modelo": [model.capitalize().replace('_', ' ') for model in aic_dict],
        "AIC": [aic_dict[model] for model in aic_dict],
        "RSS": [rss_dict[model] for model in aic_dict],
        "k": [k_dict[model] for model in aic_dict],
        "Mejor Modelo": ["Sí" if model == best_model else "No" for model in aic_dict]
    })

    latex_table = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{tabularx}

\begin{document}

\begin{table}[ht]
\centering
\caption{Comparación de Modelos según AIC}
\label{tab:model_comparison}
\begin{tabularx}{\textwidth}{l S[table-format=3.2] S[table-format=3.2] c c}
\toprule
Modelo & {AIC} & {RSS} & {k} & {Mejor Modelo} \\
\midrule
"""
    for index, row in df_aic.iterrows():
        latex_table += f"{row['Modelo']} & {row['AIC']:.2f} & {row['RSS']:.2f} & {int(row['k'])} & {row['Mejor Modelo']} \\\\ \n"

    latex_table += r"""\bottomrule
\end{tabularx}
\end{table}

\end{document}
"""

    latex_output_path = os.path.join(output_dir, "model_comparison_results.tex")
    with open(latex_output_path, "w") as f:
        f.write(latex_table)
    print(f"Tabla LaTeX de comparación de modelos generada en: {latex_output_path}")

    if compile_latex_flag:
        compile_latex(latex_output_path)

def save_all_parameters(params_dict, errors_dict, output_dir, compile_latex_flag):
    txt_output_path = os.path.join(output_dir, "model_parameters_all.txt")
    with open(txt_output_path, "w") as f:
        f.write("Modelo\tParámetros\tErrores Estándar\n")
        for model, params in params_dict.items():
            errors = errors_dict[model]
            params_str = ", ".join([f"{p:.5g}" for p in params])
            errors_str = ", ".join([f"{e:.5g}" for e in errors])
            model_name = model.capitalize().replace('_', ' ')
            f.write(f"{model_name}\t{params_str}\t{errors_str}\n")
    print(f"Parámetros de todos los modelos guardados en: {txt_output_path}")

    data = []
    for model in params_dict:
        params = params_dict[model]
        errors = errors_dict[model]
        if model == "michaelis":
            param_names = ["Vmax", "Km"]
        elif model == "hill":
            param_names = ["Vmax", "Km", "h"]
        elif model == "substrate_inhibition":
            param_names = ["Vmax", "Km", "Ki"]
        for i, param in enumerate(param_names):
            data.append({
                "Modelo": model.capitalize().replace('_', ' '),
                "Parámetro": param,
                "Valor": f"{params[i]:.5g}",
                "Error Estándar": f"{errors[i]:.5g}"
            })

    df_params = pd.DataFrame(data)

    latex_table = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{tabularx}

\begin{document}

\begin{table}[ht]
\centering
\caption{Parámetros Ajustados de los Modelos}
\label{tab:model_parameters_all}
\begin{tabularx}{\textwidth}{l l S[table-format=3.5] S[table-format=3.5]}
\toprule
Modelo & Parámetro & {Valor} & {Error Estándar} \\
\midrule
"""
    for index, row in df_params.iterrows():
        latex_table += f"{row['Modelo']} & {row['Parámetro']} & {row['Valor']} & {row['Error Estándar']} \\\\ \n"

    latex_table += r"""\bottomrule
\end{tabularx}
\end{table}

\end{document}
"""
    latex_output_path = os.path.join(output_dir, "model_parameters_all.tex")
    with open(latex_output_path, "w") as f:
        f.write(latex_table)
    print(f"Tabla LaTeX de parámetros de todos los modelos generada en: {latex_output_path}")

    if compile_latex_flag:
        compile_latex(latex_output_path)

def save_best_model_parameters(model, params, errors, output_dir, compile_latex_flag):
    model_name = model.lower().replace(' ', '_')
    txt_output_path = os.path.join(output_dir, f"{model_name}_BEST_FIT_parameters.txt")
    with open(txt_output_path, "w") as f:
        f.write(f"Modelo: {model.capitalize().replace('_', ' ')} (BEST_FIT)\n")
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
    print(f"Parámetros del mejor modelo guardados en: {txt_output_path}")

    if model == "michaelis":
        param_names = ["Vmax", "Km"]
    elif model == "hill":
        param_names = ["Vmax", "Km", "h"]
    elif model == "substrate_inhibition":
        param_names = ["Vmax", "Km", "Ki"]

    data = []
    for i, param in enumerate(param_names):
        data.append({
            "Parámetro": param,
            "Valor": f"{params[i]:.5g}",
            "Error Estándar": f"{errors[i]:.5g}"
        })

    df_best_params = pd.DataFrame(data)

    latex_table = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{tabularx}

\begin{document}

\begin{table}[ht]
\centering
\caption{Parámetros Ajustados del Modelo """ + f"{model.capitalize().replace('_', ' ')} (BEST_FIT)" + r"""}
\label{tab:""" + f"{model_name}_BEST_FIT_parameters" + r"""}
\begin{tabularx}{\textwidth}{l S[table-format=3.5] S[table-format=3.5]}
\toprule
Parámetro & {Valor} & {Error Estándar} \\
\midrule
"""
    for index, row in df_best_params.iterrows():
        latex_table += f"{row['Parámetro']} & {row['Valor']} & {row['Error Estándar']} \\\\ \n"

    latex_table += r"""\bottomrule
\end{tabularx}
\end{table}

\end{document}
"""
    latex_output_path = os.path.join(output_dir, f"{model_name}_BEST_FIT_parameters.tex")
    with open(latex_output_path, "w") as f:
        f.write(latex_table)
    print(f"Tabla LaTeX de parámetros del mejor modelo generada en: {latex_output_path}")

    if compile_latex_flag:
        compile_latex(latex_output_path)

def plot_fit_and_residuals(substrates, mean_activity, std_activity, model, params, output_dir, is_best, x_label, y_label, title):
    s_fit = np.linspace(0, max(substrates)*1.1, 500)
    if model == "michaelis":
        fitted_curve = michaelis_menten(s_fit, *params)
        model_name = "Michaelis-Menten"
        fit_filename = "michaelis_menten"
    elif model == "hill":
        fitted_curve = hill_equation(s_fit, *params)
        model_name = "Hill"
        fit_filename = "hill"
    else:
        fitted_curve = substrate_inhibition(s_fit, *params)
        model_name = "Substrate Inhibition"
        fit_filename = "substrate_inhibition"

    suffix = "_BEST_FIT" if is_best else ""

    plt.figure(figsize=(10, 8))
    plt.errorbar(substrates, mean_activity, yerr=std_activity, fmt='o', color='black', elinewidth=1)
    plt.plot(s_fit, fitted_curve, '-', color='black')
    plt.xlabel(x_label, fontsize=22, fontweight='bold')
    plt.ylabel(y_label, fontsize=22, fontweight='bold')
    plt.title(title, fontsize=22, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    fit_path = os.path.join(output_dir, f"{fit_filename}{suffix}.png")
    plt.savefig(fit_path, dpi=300)
    plt.close()
    print(f"Gráfico del modelo {model_name}{' (BEST_FIT)' if is_best else ''} guardado en: {fit_path}")

    if model == "michaelis":
        fitted = michaelis_menten(substrates, *params)
    elif model == "hill":
        fitted = hill_equation(substrates, *params)
    else:
        fitted = substrate_inhibition(substrates, *params)
    residuals = mean_activity - fitted

    plt.figure(figsize=(10, 4))
    plt.scatter(substrates, residuals, color='black')
    plt.axhline(0, linestyle='--', color='black')
    plt.xlabel(x_label, fontsize=22, fontweight='bold')
    plt.ylabel("Residual", fontsize=22, fontweight='bold')
    plt.title(f"Residuals {model_name}", fontsize=22, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    residuals_plot_path = os.path.join(output_dir, f"{fit_filename}_residuals_plot{suffix}.png")
    plt.savefig(residuals_plot_path, dpi=300)
    plt.close()
    print(f"Gráfico de residuales del modelo {model_name}{' (BEST_FIT)' if is_best else ''} guardado en: {residuals_plot_path}")

    residuals_df = pd.DataFrame({
        "substrate_concentration": substrates,
        "mean_activity": mean_activity,
        "fitted_activity": fitted,
        "residuals": residuals
    })
    residuals_csv_path = os.path.join(output_dir, f"{fit_filename}_residuals{suffix}.csv")
    residuals_df.to_csv(residuals_csv_path, index=False)
    print(f"Residuales guardados en: {residuals_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Using Akaike Information Criterion (AIC).")
    parser.add_argument("--input", type=str, required=True, help="Data route.")
    parser.add_argument("--file_type", choices=["txt", "csv"], default="txt",
                        help="File type: 'txt' or 'csv' (default: txt).")
    parser.add_argument("--sep", type=str, default=",", help="default: ','.")
    parser.add_argument("--x_label", type=str, default="[S] (Concentration units)", help="X axis label.")
    parser.add_argument("--y_label", type=str, default="Activity (U/mg)", help="Y axis label.")
    parser.add_argument("--title", type=str, default="Saturation curve", help="Graph title.")
    parser.add_argument("--compile_latex", action="store_true", default=False, help="Compile LaTeX tables to PDF")

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Leer datos
    if args.file_type == "txt":
        try:
            data = pd.read_csv(args.input, sep=args.sep, header=None, engine='python')
        except Exception as e:
            print(f"Error al leer el archivo TXT: {e}")
            sys.exit(1)
        num_cols = data.shape[1]
        if num_cols < 2:
            raise ValueError("El archivo TXT debe tener al menos 2 columnas.")
        column_names = ['substrate_concentration'] + [f'replicate_{i}' for i in range(1, num_cols)]
        data.columns = column_names
    else:  # CSV
        # Intentar leer con encabezado
        try:
            data = pd.read_csv(args.input, sep=args.sep, engine='python')
            if 'substrate_concentration' not in data.columns:
                # Si no está la columna, leer sin encabezado
                data = pd.read_csv(args.input, sep=args.sep, header=None, engine='python')
                num_cols = data.shape[1]
                if num_cols < 2:
                    raise ValueError("El archivo CSV sin encabezado debe tener al menos 2 columnas.")
                column_names = ['substrate_concentration'] + [f'replicate_{i}' for i in range(1, num_cols)]
                data.columns = column_names
        except Exception as e:
            # Si falla con encabezado, intentar sin encabezado
            try:
                data = pd.read_csv(args.input, sep=args.sep, header=None, engine='python')
                num_cols = data.shape[1]
                if num_cols < 2:
                    raise ValueError("El archivo CSV sin encabezado debe tener al menos 2 columnas.")
                column_names = ['substrate_concentration'] + [f'replicate_{i}' for i in range(1, num_cols)]
                data.columns = column_names
            except Exception as ee:
                print(f"Error al leer el archivo CSV: {ee}")
                sys.exit(1)

    replicate_cols = [col for col in data.columns if col.startswith('replicate_')]
    if len(replicate_cols) == 0:
        raise ValueError("Debe haber al menos una columna de replicados (replicate_).")

    data["mean_activity"] = data[replicate_cols].mean(axis=1, skipna=True)
    data["std_activity"] = data[replicate_cols].std(axis=1, skipna=True)
    data = data.dropna(subset=replicate_cols, how='all')

    substrate_concentration = data["substrate_concentration"].values
    mean_activity = data["mean_activity"].values
    std_activity = data["std_activity"].values

    models = ["michaelis", "hill", "substrate_inhibition"]
    aic_results = {}
    rss_results = {}
    k_results = {}
    fitted_params = {}
    fitted_errors = {}

    for model in models:
        fit = fit_model(model, substrate_concentration, mean_activity)
        params = fit["params"]
        errors = fit["errors"]
        fitted_params[model] = params
        fitted_errors[model] = errors

        if np.isnan(params).any():
            aic = np.inf
            rss = np.inf
            k = len(params)
            print(f"No se pudo calcular AIC para el modelo {model.capitalize().replace('_', ' ')} debido a un ajuste fallido.")
        else:
            if model == "michaelis":
                fitted = michaelis_menten(substrate_concentration, *params)
            elif model == "hill":
                fitted = hill_equation(substrate_concentration, *params)
            else:
                fitted = substrate_inhibition(substrate_concentration, *params)
            residuals = mean_activity - fitted
            rss = np.sum(residuals**2)
            n = len(mean_activity)
            k = len(params)
            aic = calculate_aic(n, rss, k)
        aic_results[model] = aic
        rss_results[model] = rss
        k_results[model] = k

    valid_aic = {m: aic for m, aic in aic_results.items() if not np.isinf(aic)}
    if not valid_aic:
        print("No se pudo ajustar ninguno de los modelos.")
        sys.exit(1)

    best_model = min(valid_aic, key=valid_aic.get)
    best_aic = valid_aic[best_model]
    print(f"Mejor modelo según AIC: {best_model.capitalize().replace('_', ' ')} con AIC = {best_aic:.2f}")

    save_aic_results(aic_results, rss_results, k_results, best_model, script_dir, args.compile_latex)
    save_all_parameters(fitted_params, fitted_errors, script_dir, args.compile_latex)

    # Graficar todas las curvas y residuales
    for model in models:
        params = fitted_params[model]
        if np.isnan(params).any():
            continue
        is_best = (model == best_model)
        plot_fit_and_residuals(substrate_concentration, mean_activity, std_activity, model, params, script_dir, is_best, args.x_label, args.y_label, args.title)

    best_params = fitted_params[best_model]
    best_errors = fitted_errors[best_model]
    if np.isnan(best_params).any():
        print(f"No se pudieron generar tablas para el mejor modelo {best_model.capitalize().replace('_', ' ')}.")
        sys.exit(1)
    save_best_model_parameters(best_model, best_params, best_errors, script_dir, args.compile_latex)

    print("Análisis completo.")

if __name__ == "__main__":
    main()
