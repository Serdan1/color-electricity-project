import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# === 1. Cargar datos de electricidad ===
def load_electricity_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Renombrar la columna de año si existe
    if "Year" in df.columns:
        df = df.rename(columns={"Year": "year"})
    
    # Buscar la columna que contenga "electricity" o "consumption"
    value_col = None
    for col in df.columns:
        if "electricity" in col.lower() or "consumption" in col.lower():
            value_col = col
            break
    
    if not value_col:
        raise ValueError("❌ No se encontró una columna de consumo eléctrico en el CSV.")
    
    # Nos quedamos SOLO con 'year' y la columna numérica de consumo
    df = df[["year", value_col]].rename(columns={value_col: "electricity"})
    
    # Convertimos a numérico (eliminamos texto)
    df["electricity"] = pd.to_numeric(df["electricity"], errors="coerce")
    
    # Eliminamos filas sin número
    df = df.dropna(subset=["year", "electricity"])
    
    # Convertir año a int
    df["year"] = df["year"].astype(int)
    
    # Agrupamos (si hay duplicados de año)
    df = df.groupby("year", as_index=False)["electricity"].mean()
    
    print(f"✅ Datos cargados correctamente: {len(df)} años entre {df['year'].min()} y {df['year'].max()}")
    return df


# === 2. Calcular métricas de color por carpeta ===
def analyze_colors(root_folder):
    results = []
    for decade_folder in sorted(Path(root_folder).glob("*")):
        if decade_folder.is_dir():
            images = list(decade_folder.glob("*.jpg")) + list(decade_folder.glob("*.png"))
            sat_list = []
            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                img = np.array(img) / 255.0
                mx = img.max(axis=2)
                mn = img.min(axis=2)
                sat = np.mean((mx - mn) / (mx + 1e-6))
                sat_list.append(sat)
            if len(sat_list) > 0:
                year = int(decade_folder.name)
                results.append({"year": year, "saturation": np.mean(sat_list)})
    return pd.DataFrame(results)

# === 3. Unir datasets y visualizar ===
def main():
    elec = load_electricity_data("data/electricity.csv")
    color = analyze_colors("data/colors")

    merged = pd.merge_asof(color.sort_values("year"), elec.sort_values("year"), on="year")
    merged.dropna(inplace=True)

    # Guardar dataset combinado
    merged.to_csv("data/color_electricity_combined.csv", index=False)

    # Visualización
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(merged["year"], merged["saturation"], color="tab:blue", label="Saturación media")
    ax2 = ax1.twinx()
    ax2.plot(merged["year"], merged["electricity"], color="tab:red", label="Consumo eléctrico")
    ax1.set_xlabel("Año")
    ax1.set_ylabel("Saturación media (colores)")
    ax2.set_ylabel("Consumo eléctrico per cápita (kWh)")
    plt.title("Evolución del color y el consumo eléctrico (1880–2025)")
    plt.savefig("outputs/fase0_color_electricity.png", dpi=150)
    plt.show()

    print("\n✅ Dataset combinado guardado en data/color_electricity_combined.csv")

if __name__ == "__main__":
    main()
