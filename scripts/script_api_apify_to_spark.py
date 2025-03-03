import os
import re
import json
import requests
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import udf, col, lit

# =============================================================================
# PARTIE ML – Prédiction du genre avec gender_guesser
# =============================================================================

import gender_guesser.detector as gender
d = gender.Detector()

def guess_gender_best(full_name, username):
    """
    Prédit le genre en se basant sur 'full_name' et 'username'.
    - full_name : on prend le premier mot (supposé être le prénom) et on attribue
      une confiance élevée (0.9) si la prédiction est claire.
    - username : on nettoie la chaîne (on garde uniquement les lettres) et on attribue
      une confiance moindre (0.7).
    Retourne (predicted_gender, confidence).
    """
    # Prédiction basée sur full_name
    if full_name and full_name.strip():
        first_name = full_name.split()[0]
        pred_full = d.get_gender(first_name)
        if pred_full in ["male", "mostly_male"]:
            gender_full = "male"
            conf_full = 0.9
        elif pred_full in ["female", "mostly_female"]:
            gender_full = "female"
            conf_full = 0.9
        else:
            gender_full = "unknown"
            conf_full = 0.5
    else:
        gender_full = "unknown"
        conf_full = 0.0

    # Prédiction basée sur username
    if username and username.strip():
        cleaned_username = re.sub(r'[^A-Za-z]', '', username)
        if cleaned_username:
            pred_user = d.get_gender(cleaned_username)
            if pred_user in ["male", "mostly_male"]:
                gender_user = "male"
                conf_user = 0.7
            elif pred_user in ["female", "mostly_female"]:
                gender_user = "female"
                conf_user = 0.7
            else:
                gender_user = "unknown"
                conf_user = 0.4
        else:
            gender_user = "unknown"
            conf_user = 0.0
    else:
        gender_user = "unknown"
        conf_user = 0.0

    # On sélectionne la meilleure prédiction
    if conf_full >= conf_user:
        return (gender_full, float(conf_full))
    else:
        return (gender_user, float(conf_user))

# On déclare l'UDF pour Spark avec un schéma structuré
gender_udf = udf(
    guess_gender_best,
    StructType([
        StructField("predicted_gender", StringType(), True),
        StructField("confidence", FloatType(), True)
    ])
)

# =============================================================================
# INITIALISATION DE SPARK
# =============================================================================

spark = SparkSession.builder \
    .appName("ApifyInstagramDataProcessing") \
    .config("spark.jars", "jars/postgresql-42.2.27.jar") \
    .getOrCreate()

# =============================================================================
# DÉFINITION DES PARAMÈTRES ET NORMALISATION DU COMPTE À SCRAPER
# =============================================================================

account = "USERNAME_PLACEHOLDER"  # Ce placeholder sera remplacé par le DAG
normalized_account = account.replace(".", "-").replace("_", "-")

# =============================================================================
# CONFIGURATION DE L'URL DE L'API
# =============================================================================

api_token = "apify_api_1F6nxXD6wiqCh2puFAa5JF6lILz3CL3nzeuU"
url = (
    f"https://api.apify.com/v2/actor-tasks/bytacticxhd~insta-following-{normalized_account}"
    f"/run-sync-get-dataset-items?token={api_token}"
)

print(f"🌀 [INFO] Lancement du script pour le compte : {account} (normalisé : {normalized_account})")
print(f"🌀 [INFO] Appel API : {url}")

# =============================================================================
# APPEL À L'API ET RÉCUPÉRATION DES DONNÉES
# =============================================================================

response = requests.get(url)
print(f"🌀 [INFO] Statut de l'API : {response.status_code}")

current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime("%H%M")

data = None

# =============================================================================
# STOCKAGE DES DONNÉES RAW
# =============================================================================

raw_layer = "data/raw"
raw_group = "apify"
raw_table_name = f"apify_instagram_data_{normalized_account}"
raw_filename = "raw.json"
raw_output_path = os.path.join(raw_layer, raw_group, raw_table_name, current_date)
os.makedirs(raw_output_path, exist_ok=True)
raw_file_path = os.path.join(raw_output_path, raw_filename)

if response.status_code in (200, 201):
    data = response.json()
    with open(raw_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ [INFO] Données RAW enregistrées dans '{raw_file_path}'.")
else:
    print(f"❌ [ERREUR] API status={response.status_code}, pas de données RAW enregistrées.")

# =============================================================================
# TRANSFORMATION DES DONNÉES
# =============================================================================
if data:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Soit data["items"], soit data
        items = data.get("items", data)
    else:
        items = []

    print(f"🔎 [INFO] Nombre d'éléments récupérés pour transformation : {len(items)}")
    df_pd = pd.DataFrame(items)
    if not df_pd.empty:
        columns_kept = ["username", "full_name"]
        df_pd = df_pd[columns_kept]
    else:
        print("💡 [INFO] DataFrame vide, aucune colonne filtrée.")

    schema = StructType([
        StructField("username", StringType(), True),
        StructField("full_name", StringType(), True)
    ])
    df_spark = spark.createDataFrame(df_pd, schema=schema)

    row_count = df_spark.count()
    print(f"🔎 [INFO] Nombre de lignes dans df_spark : {row_count}")
    df_spark.show(5, truncate=False)

    # =============================================================================
    # APPLICATION DU MODÈLE ML
    # =============================================================================
    df_with_ml = df_spark.withColumn("gender_info", gender_udf(col("full_name"), col("username")))
    df_with_ml = df_with_ml \
        .withColumn("predicted_gender", col("gender_info.predicted_gender")) \
        .withColumn("confidence", col("gender_info.confidence")) \
        .drop("gender_info")

    print("🔎 [INFO] Exemples après application du modèle ML :")
    df_with_ml.show(5, truncate=False)

    # =============================================================================
    # STOCKAGE FORMATTED
    # =============================================================================
    formatted_layer = "data/formatted"
    formatted_group = "apify"
    formatted_table_name = f"apify_instagram_data_{normalized_account}"
    formatted_filename = "formatted_parquet_with_ML.parquet"
    formatted_output_path = os.path.join(formatted_layer, formatted_group, formatted_table_name, current_date)
    os.makedirs(formatted_output_path, exist_ok=True)
    formatted_parquet_file = os.path.join(formatted_output_path, formatted_filename)

    df_with_ml.write.mode("append").parquet(formatted_parquet_file)
    print(f"✅ [INFO] Données formatées => '{formatted_parquet_file}'.")

    # =============================================================================
    # STOCKAGE USAGE (TABLEAU FINAL)
    # =============================================================================
    usage_layer = "data/usage"
    usage_group = "apify"
    usage_table_name = f"apify_instagram_data_{normalized_account}"
    usage_filename = "formatted_parquet_with_ML.parquet"
    usage_output_path = os.path.join(usage_layer, usage_group, usage_table_name, current_date, current_time)
    os.makedirs(usage_output_path, exist_ok=True)
    usage_parquet_file = os.path.join(usage_output_path, usage_filename)

    df_with_ml.write.mode("overwrite").parquet(usage_parquet_file)
    print(f"✅ [INFO] Données usage final => '{usage_parquet_file}' ({df_with_ml.count()} lignes).")

    # =============================================================================
    # COMPARAISON AVEC LA DERNIÈRE EXÉCUTION
    # =============================================================================
    base_usage_path = os.path.join(usage_layer, usage_group, usage_table_name, current_date)
    df_prev = None
    previous_run = None

    if os.path.exists(base_usage_path):
        dirs = [d for d in os.listdir(base_usage_path) if os.path.isdir(os.path.join(base_usage_path, d))]
        dirs_sorted = sorted(dirs)
        for d in dirs_sorted:
            if d < current_time:
                previous_run = d

        if previous_run:
            prev_usage_file = os.path.join(base_usage_path, previous_run, usage_filename)
            try:
                df_prev = spark.read.parquet(prev_usage_file)
                print(f"✅ [INFO] Données usage précédentes chargées depuis '{prev_usage_file}'.")
            except Exception as e:
                print(f"❌ [ERREUR] Lecture usage précédente: {e}")
                df_prev = None
        else:
            print("💡 [INFO] Aucune exécution précédente trouvée pour comparaison.")
    else:
        print("💡 [INFO] Pas de dossier usage pour ce jour, pas de comparaison possible.")

    if df_prev is not None:
        df_current_sel = df_with_ml.select("username", "full_name", "predicted_gender", "confidence")
        df_prev_sel = df_prev.select("username", "full_name", "predicted_gender", "confidence")

        df_added = (df_current_sel
                    .join(df_prev_sel, on=["username", "full_name"], how="leftanti")
                    .withColumn("change", lit("added")))
        df_deleted = (df_prev_sel
                      .join(df_current_sel, on=["username", "full_name"], how="leftanti")
                      .withColumn("change", lit("deleted")))
        df_comparatif = df_added.unionByName(df_deleted)

        print("🔎 [INFO] Tableau comparatif (10 lignes) :")
        df_comparatif.show(10, truncate=False)

        # Écriture du COMPARATIF en Parquet
        comparatif_filename = "comparatif_parquet_with_ML.parquet"
        comparatif_parquet_file = os.path.join(usage_output_path, comparatif_filename)

        df_comparatif.write.mode("overwrite").parquet(comparatif_parquet_file)
        print(f"✅ [INFO] Données comparatives => '{comparatif_parquet_file}' ({df_comparatif.count()} lignes).")
    else:
        print("💡 [INFO] Pas de comparaison effectuée (pas d'exécution précédente).")

    # =============================================================================
    # POSTGRESQL (OPTIONNEL) — avec correction du nom de table
    # =============================================================================
    table_name_for_db = formatted_table_name.replace("-", "_")
    # Cela supprime le tiret pour éviter "syntax error at or near '-'"

    try:
        df_postgres = spark.read \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://127.0.0.1:5432/airflow") \
            .option("dbtable", table_name_for_db) \
            .option("user", "airflow") \
            .option("password", "airflow") \
            .option("driver", "org.postgresql.Driver") \
            .load()
        print("🔎 [INFO] Lecture depuis PostgreSQL (10 lignes) :")
        df_postgres.show(10, truncate=False)
    except Exception as e:
        print(f"❌ [ERREUR] Lecture PostgreSQL: {e}")

    try:
        df_with_ml.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://127.0.0.1:5432/airflow") \
            .option("dbtable", table_name_for_db) \
            .option("user", "airflow") \
            .option("password", "airflow") \
            .option("driver", "org.postgresql.Driver") \
            .mode("append") \
            .save()
        print("✅ [INFO] Données écrites dans PostgreSQL (append), table:", table_name_for_db)
    except Exception as e:
        print(f"❌ [ERREUR] Écriture PostgreSQL : {e}")

else:
    print("💡 [INFO] Aucune donnée renvoyée par l'API, pas de transformation possible.")

spark.stop()
print(f"🔚 [INFO] Fin du script pour {account} (normalisé: {normalized_account}).")
