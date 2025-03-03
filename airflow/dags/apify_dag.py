from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
import re
import subprocess
import glob

# -------------------------------------------------------------------------
# Configuration et constantes
# -------------------------------------------------------------------------

BASE_DIR = "/root/mlecorre/datalake_final"  # À adapter selon ta config
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
GENERATED_DIR = os.path.join(BASE_DIR, "generated_scripts")
ACCOUNTS_FILE = os.path.join(SCRIPTS_DIR, "instagram_accounts_to_scrape.txt")
BASE_SCRIPT = os.path.join(SCRIPTS_DIR, "script_api_apify_to_spark.py")
JARS_PATH = os.path.join(BASE_DIR, "jars", "postgresql-42.2.27.jar")  # Jar PostgreSQL
ES_SPARK_JAR = os.path.join(BASE_DIR, "jars", "elasticsearch-spark-30_2.12-8.5.3.jar")  # Jar ES

# Exemple d'ancien snapshot pour comparaison globale (optionnel)
OLD_PARQUET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "usage_to_combined",
    "apify",
    "apify_instagram_data",
    "20250228",
    "1243",
    "final_aggregated.parquet"
)

default_args = {
    'owner': 'votre_nom',
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'apify_dynamic_pipeline',
    default_args=default_args,
    schedule_interval='@hourly',  # ou la crontab de ton choix
    catchup=False
)

# -------------------------------------------------------------------------
# Tâche 1 : GÉNÉRATION DES SCRIPTS (TaskFlow)
# -------------------------------------------------------------------------
@task
def generate_scripts():
    """
    Lit le fichier ACCOUNTS_FILE et, pour chaque username, génère un script
    en remplaçant 'USERNAME_PLACEHOLDER' par la version normalisée.
    Renvoie une liste de dictionnaires contenant 'account', 'normalized' et 'script_path'.
    """
    with open(BASE_SCRIPT, "r", encoding="utf-8") as f:
        base_content = f.read()
    with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
        accounts = [line.strip() for line in f if line.strip()]
    generated = []
    os.makedirs(GENERATED_DIR, exist_ok=True)
    for account in accounts:
        normalized = account.replace(".", "-").replace("_", "-")
        new_content = base_content.replace("USERNAME_PLACEHOLDER", normalized)
        script_name = f"Script_api_apify_{normalized}.py"
        script_path = os.path.join(GENERATED_DIR, script_name)
        with open(script_path, "w", encoding="utf-8") as f_out:
            f_out.write(new_content)
        generated.append({
            "account": account,
            "normalized": normalized,
            "script_path": script_path
        })
    print("✅ [generate_scripts] Scripts générés :", generated)
    return generated

# -------------------------------------------------------------------------
# Tâche 2 : EXÉCUTION DES SCRIPTS EN PARALLÈLE (TaskFlow avec Mapping)
# -------------------------------------------------------------------------
@task
def run_single_script(script_info: dict):
    """
    Exécute un script via spark-submit en utilisant le chemin contenu dans script_info.
    """
    command = f"spark-submit --jars {JARS_PATH} {script_info['script_path']}"
    print(f"🌀 [run_single_script] Exécution : {command}")
    subprocess.run(command, shell=True, check=True)
    return script_info

# -------------------------------------------------------------------------
# Tâche 3 : AGRÉGATION DES RÉSULTATS + CHARGEMENT EN BDD (PythonOperator)
# -------------------------------------------------------------------------
def aggregate_results(**kwargs):
    """
    1) Lit pour chaque compte :
         - "formatted_parquet_with_ML.parquet" (tables finales)
         - "comparatif_parquet_with_ML.parquet" (tables comparatives) si elles existent
    2) Agrège et écrit deux Parquets dans usage_to_combined :
         - final_aggregated.parquet
         - final_comparatif.parquet
    3) Compare le final agrégé avec OLD_PARQUET_PATH si présent.
    4) Insère ces Parquets dans PostgreSQL.
    5) Push les chemins des Parquets via XCom pour la tâche d'indexation.
    """
    import glob
    import os
    from datetime import datetime
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import lit

    spark = SparkSession.builder \
        .appName("Aggregation") \
        .config("spark.jars", JARS_PATH) \
        .getOrCreate()

    ti = kwargs['ti']
    generated = ti.xcom_pull(key="return_value", task_ids="generate_scripts")
    if not generated:
        print("💡 [aggregate_results] Aucune donnée à agréger (pas de scripts générés).")
        spark.stop()
        return

    aggregated_final_df = None
    aggregated_comp_df = None
    current_date = datetime.now().strftime("%Y%m%d")
    current_time = datetime.now().strftime("%H%M")

    # Parcours des comptes pour lire "final" et "comparatif"
    for info in generated:
        norm = info["normalized"]
        usage_final_pattern = os.path.join(
            BASE_DIR, "data", "usage", "apify",
            f"apify_instagram_data_{norm}",
            current_date,
            "*",
            "formatted_parquet_with_ML.parquet"
        )
        final_files = glob.glob(usage_final_pattern)
        if final_files:
            for fpath in final_files:
                print(f"🔎 [aggregate_results] Lecture FINAL {fpath} pour {norm}")
                try:
                    df_final = spark.read.parquet(fpath)
                    df_final = df_final.withColumn("username_scraped", lit(norm))
                    aggregated_final_df = df_final if aggregated_final_df is None else aggregated_final_df.unionByName(df_final)
                except Exception as e:
                    print(f"❌ [aggregate_results] Erreur lecture finale {fpath} : {e}")
        else:
            print(f"💡 [aggregate_results] Aucune table finale trouvée pour {norm}")

        usage_comp_pattern = os.path.join(
            BASE_DIR, "data", "usage", "apify",
            f"apify_instagram_data_{norm}",
            current_date,
            "*",
            "comparatif_parquet_with_ML.parquet"
        )
        comp_files = glob.glob(usage_comp_pattern)
        if comp_files:
            for fpath in comp_files:
                print(f"🔎 [aggregate_results] Lecture COMPARATIF {fpath} pour {norm}")
                try:
                    df_comp = spark.read.parquet(fpath)
                    df_comp = df_comp.withColumn("username_scraped", lit(norm))
                    aggregated_comp_df = df_comp if aggregated_comp_df is None else aggregated_comp_df.unionByName(df_comp)
                except Exception as e:
                    print(f"❌ [aggregate_results] Erreur lecture comparatif {fpath} : {e}")
        else:
            print(f"💡 [aggregate_results] Aucune table comparatif trouvée pour {norm}")

    # Écriture des fichiers agrégés
    combined_path = os.path.join(
        BASE_DIR, "data", "usage_to_combined", "apify",
        "apify_instagram_data", current_date, current_time
    )
    os.makedirs(combined_path, exist_ok=True)
    final_aggregated_file = os.path.join(combined_path, "final_aggregated.parquet")
    final_comp_file = os.path.join(combined_path, "final_comparatif.parquet")

    if aggregated_final_df is not None:
        aggregated_final_df.write.mode("append").parquet(final_aggregated_file)
        print(f"✅ [aggregate_results] final_aggregated.parquet => {final_aggregated_file}")
        aggregated_final_df.show(10, truncate=False)
    else:
        print("💡 [aggregate_results] Aucun DF final agrégé à écrire.")

    if aggregated_comp_df is not None:
        aggregated_comp_df.write.mode("append").parquet(final_comp_file)
        print(f"✅ [aggregate_results] final_comparatif.parquet => {final_comp_file}")
        aggregated_comp_df.show(10, truncate=False)
    else:
        print("💡 [aggregate_results] Aucun DF comparatif agrégé à écrire.")

    # Comparaison globale (optionnel)
    if aggregated_final_df is not None and os.path.exists(OLD_PARQUET_PATH):
        print("📌 [aggregate_results] Ancien snapshot détecté, comparaison globale en cours ...")
        try:
            oldDF = spark.read.parquet(OLD_PARQUET_PATH)
            join_cols = ["username", "full_name"]
            added = aggregated_final_df.join(oldDF, join_cols, "left_anti").withColumn("change", lit("added_global"))
            deleted = oldDF.join(aggregated_final_df, join_cols, "left_anti").withColumn("change", lit("deleted_global"))
            global_compDF = added.unionByName(deleted)
            global_comp_file = os.path.join(combined_path, "final_global_comparatif.parquet")
            global_compDF.write.mode("append").parquet(global_comp_file)
            print(f"✅ [aggregate_results] final_global_comparatif.parquet => {global_comp_file}")
            global_compDF.show(20, truncate=False)
        except Exception as e:
            print(f"❌ [aggregate_results] Erreur comparaison globale : {e}")
    else:
        print("🛑 [aggregate_results] Pas d'ancien snapshot ou pas de DF final, pas de comparaison globale.")

    # Insertion en PostgreSQL
    try:
        df_aggreg = spark.read.parquet(final_aggregated_file)
        print("🔵 [aggregate_results] Insertion du Parquet final_aggregated dans PostgreSQL ...")
        df_aggreg.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://127.0.0.1:5432/airflow") \
            .option("dbtable", "final_aggregated_usage") \
            .option("user", "airflow") \
            .option("password", "airflow") \
            .option("driver", "org.postgresql.Driver") \
            .mode("append") \
            .save()
        print("✅ [aggregate_results] final_aggregated.parquet inséré dans 'final_aggregated_usage'.")
    except Exception as e:
        print(f"❌ [aggregate_results] Erreur lors de l'insertion final_aggregated : {e}")

    try:
        df_comp = spark.read.parquet(final_comp_file)
        print("🔵 [aggregate_results] Insertion du Parquet final_comparatif dans PostgreSQL ...")
        df_comp.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://127.0.0.1:5432/airflow") \
            .option("dbtable", "final_comparatif_usage") \
            .option("user", "airflow") \
            .option("password", "airflow") \
            .option("driver", "org.postgresql.Driver") \
            .mode("append") \
            .save()
        print("✅ [aggregate_results] final_comparatif.parquet inséré dans 'final_comparatif_usage'.")
    except Exception as e:
        print(f"❌ [aggregate_results] Erreur lors de l'insertion final_comparatif : {e}")

    # Push des chemins pour la tâche 4
    ti.xcom_push(key="final_aggregated_file", value=final_aggregated_file)
    ti.xcom_push(key="final_comparatif_file", value=final_comp_file)

    spark.stop()

aggregate_task = PythonOperator(
    task_id="aggregate_results",
    python_callable=aggregate_results,
    provide_context=True,
    dag=dag
)

# -------------------------------------------------------------------------
# Tâche 4 : INDEXATION ELASTICSEARCH (PythonOperator)
# -------------------------------------------------------------------------
def index_to_elasticsearch(**kwargs):
    """
    Lit final_aggregated.parquet et final_comparatif.parquet (via XCom)
    et les envoie dans Elasticsearch.
    IMPORTANT : On inclut à la fois le jar PostgreSQL et le jar Elasticsearch.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import lit
    from datetime import datetime

    ti = kwargs['ti']
    final_aggregated_file = ti.xcom_pull(task_ids="aggregate_results", key="final_aggregated_file")
    final_comparatif_file = ti.xcom_pull(task_ids="aggregate_results", key="final_comparatif_file")

    if not final_aggregated_file or not final_comparatif_file:
        print("💡 [index_to_elasticsearch] Pas de chemins parquet reçus, indexation annulée.")
        return

    # Concaténation des jars PostgreSQL et Elasticsearch-Spark
    jars_list = f"{JARS_PATH},{ES_SPARK_JAR}"
    spark = SparkSession.builder \
        .appName("IndexationElasticsearch") \
        .config("spark.jars", jars_list) \
        .getOrCreate()

    try:
        df_final = spark.read.parquet(final_aggregated_file)
        df_final = df_final.withColumn("indexed_at", lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # On retire le type de document de l'option es.resource pour éviter l'erreur
        df_final.write \
            .format("org.elasticsearch.spark.sql") \
            .option("es.nodes", "localhost") \
            .option("es.port", "9200") \
            .option("es.nodes.wan.only", "true") \
            .option("es.resource", "final_aggregated_index") \
            .option("es.mapping.id", "username") \
            .mode("overwrite") \
            .save()
        print("✅ [Elasticsearch] final_aggregated.parquet indexé dans 'final_aggregated_index'.")
    except Exception as e:
        print(f"❌ [Elasticsearch] Erreur indexation final_aggregated : {e}")

    try:
        df_comp = spark.read.parquet(final_comparatif_file)
        df_comp = df_comp.withColumn("indexed_at", lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        df_comp.write \
            .format("org.elasticsearch.spark.sql") \
            .option("es.nodes", "localhost") \
            .option("es.port", "9200") \
            .option("es.nodes.wan.only", "true") \
            .option("es.resource", "final_aggregated_index") \
            .option("es.mapping.id", "username") \
            .mode("overwrite") \
            .save()
        print("✅ [Elasticsearch] final_comparatif.parquet indexé dans 'final_comparatif_index'.")
    except Exception as e:
        print(f"❌ [Elasticsearch] Erreur indexation final_comparatif : {e}")

    spark.stop()

index_task = PythonOperator(
    task_id="index_to_elasticsearch",
    python_callable=index_to_elasticsearch,
    provide_context=True,
    dag=dag
)

# -------------------------------------------------------------------------
# ORDONNANCEMENT (dans un bloc with dag: pour s'assurer que tous les tasks sont associés au DAG)
# -------------------------------------------------------------------------
with dag:
    generated_scripts = generate_scripts()
    run_all_scripts = run_single_script.expand(script_info=generated_scripts)
    generated_scripts >> run_all_scripts >> aggregate_task >> index_task
