from datetime import datetime, timezone
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, IntegerType, LongType, ShortType, DoubleType, FloatType, BooleanType, DateType, TimestampType, BinaryType, StructField, ArrayType
from pyspark.sql.functions import count, when, isnull


class SparkUtils:

    def __init__(self, app_name, master_url=None, spark_jars=None, spark_packages=None):
        spark_builder = (
            SparkSession.builder
            .appName(app_name))

        if master_url is not None:
            spark_builder = spark_builder.master(master_url)

        if spark_jars is not None:
            spark_builder = spark_builder.config("spark.jars", spark_jars)

        if spark_packages is not None:
            spark_builder = spark_builder.config("spark.jars.packages", spark_packages)

        self._spark = spark_builder.getOrCreate()

    @property
    def spark(self):
        return self._spark

    @staticmethod
    def generate_schema(columns_info) -> StructType:
        """
        Generates a list of StructField objects from a list of tuples.

        Args:
            column_info (list of tuples): Each tuple contains (column_name, data_type_string).

        Returns:
            list: A list of StructField objects.
        """
        # Mapping from string type names to PySpark data types
        type_mapping = {
            "string": StringType(),
            "int": IntegerType(),
            "long": LongType(),
            "short": ShortType(),
            "double": DoubleType(),
            "float": FloatType(),
            "boolean": BooleanType(),
            "date": DateType(),
            "timestamp": TimestampType(),
            "binary": BinaryType(),
            "array_int": ArrayType(IntegerType()),
            "array_string": ArrayType(StringType()),
            "struct": StructType()
        }

        struct_fields = []
        for column_info in columns_info:
            col_name  = column_info[0]
            col_type  = column_info[1]

            if col_type == "struct":
                # Expect a third element with the sub-field definitions
                if len(column_info) < 3 or not isinstance(column_info[2], list):
                    raise ValueError(
                        f"Column '{col_name}' is declared as 'struct' but "
                        f"no sub-fields list was provided as the third element."
                    )
                # Recurse to build the nested StructType
                nested_schema = SparkUtils.generate_schema(column_info[2])
                struct_field  = StructField(col_name, nested_schema, True)

            elif col_type not in type_mapping:
                raise ValueError(f"Unsupported data type: '{col_type}' for column '{col_name}'")

            else:
                # Create a StructField for the column
                struct_field = StructField(column_info[0], type_mapping[column_info[1]], True)

            struct_fields.append(struct_field)

        return StructType(struct_fields)

    @staticmethod
    def count_nulls(df):
        return df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])

    # ------------------------------------------------------------------
    # Debugging helpers (useful on EMR, where stdout is buried in logs)
    # ------------------------------------------------------------------

    @staticmethod
    def schema_as_text(df) -> str:
        """Returns the same tree that df.printSchema() prints, as a string."""
        return df._jdf.schema().treeString()

    @staticmethod
    def sample_as_text(df, rows=20, truncate=80, vertical=True) -> str:
        """Returns the same table that df.show() prints, as a string."""
        return df._jdf.showString(rows, truncate, vertical)

    @staticmethod
    def sample_as_json(df, rows=20) -> list:
        """Returns the first rows as JSON strings (keeps nested structs readable)."""
        return df.limit(rows).toJSON().collect()

    def debug_report(self, df, title="", rows=20, count_rows=False) -> str:
        """Builds a text report with input files, schema and a sample of records."""
        sample = df.limit(rows)
        files = df.inputFiles()

        lines = [f"Debug report - {datetime.now(timezone.utc).isoformat()}", title, ""]

        lines.append(f"Files read: {len(files)}")
        for f in files[:10]:
            lines.append(f"  {f}")
        if len(files) > 10:
            lines.append(f"  ... and {len(files) - 10} more")
        lines.append(f"Columns: {len(df.columns)}")
        if count_rows:
            lines.append(f"Row count: {df.count():,}")  # full scan
        lines.append("")

        lines.append("=== SCHEMA ===")
        lines.append(SparkUtils.schema_as_text(df))

        lines.append(f"=== SAMPLE ({rows} rows, vertical) ===")
        lines.append(SparkUtils.sample_as_text(sample, rows))

        lines.append(f"=== SAMPLE ({rows} rows, JSON) ===")
        lines.extend(SparkUtils.sample_as_json(sample, rows))

        return "\n".join(lines) + "\n"

    def write_text(self, path, text):
        """
        Writes a single text file using Hadoop's FileSystem API.
        Works with s3://, hdfs:// and file:// paths (overwrites if it exists).
        """
        jvm = self._spark._jvm
        hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
        fs = hadoop_path.getFileSystem(self._spark._jsc.hadoopConfiguration())
        stream = fs.create(hadoop_path, True)
        try:
            stream.write(bytearray(text.encode("utf-8")))
        finally:
            stream.close()

    # ==================================================================
    # Proyecto final - Overture Maps, theme=buildings / type=building
    #
    # Pregunta: ¿cómo se distribuye la huella construida por región
    # (clase de edificio, altura y fuente de los datos) y qué tan completa
    # es la información de altura y nombre en cada región?
    #
    # Todos los métodos de esta sección son TRANSFORMACIONES (lazy):
    # ninguno dispara un action. La única escritura está en
    # write_partitioned(), que se llama al final del main().
    # ==================================================================

    # Altura promedio por piso (m), para estimar la altura cuando sólo hay num_floors
    METERS_PER_FLOOR = 3.0

    def read_buildings(self, path):
        """Lee los Parquet de Overture (Spark infiere el esquema del footer Parquet)."""
        return self._spark.read.parquet(path)

    @staticmethod
    def select_building_columns(df):
        """
        Proyección temprana: columnas planas y sólo los subcampos anidados que se usan.
        Con nested schema pruning, Parquet lee únicamente names.primary y sources.dataset
        en lugar de los structs/arrays completos (names.rules, sources.*), lo que reduce
        mucho la lectura, la memoria por tarea y el tamaño del shuffle de dropDuplicates.
        """
        return df.select(
            "id", "bbox", "subtype", "class", "height", "num_floors",
            "is_underground", "has_parts",
            F.col("names.primary").alias("name_primary"),
            F.expr("try_element_at(sources, 1).dataset").alias("primary_source"),
            F.size("sources").alias("source_count"),
        )

    @staticmethod
    def clean_buildings(df, dedup=False):
        """
        Data cleaning:
          1. Elimina registros malformados: sin id, sin bbox o con bbox inválido.
          2. (opcional, dedup=True) Elimina duplicados por id. Los ids de Overture (GERS)
             son únicos por release, y dropDuplicates obliga a un shuffle + sort de todo
             el dataset, así que por defecto no se aplica.
          3. Invalida (pone en null) alturas y pisos fuera de rango físico.
          4. Rellena subtype/class nulos con 'unknown'.
        """
        height_ok = (F.col("height") > 0) & (F.col("height") < 1000)       # el edificio más alto mide ~830 m
        floors_ok = (F.col("num_floors") > 0) & (F.col("num_floors") < 200)
        bbox_ok = (
            F.col("bbox.xmin").between(-180, 180) & F.col("bbox.xmax").between(-180, 180)
            & F.col("bbox.ymin").between(-90, 90) & F.col("bbox.ymax").between(-90, 90)
            & (F.col("bbox.xmin") <= F.col("bbox.xmax")) & (F.col("bbox.ymin") <= F.col("bbox.ymax"))
        )

        df = df.filter(F.col("id").isNotNull() & F.col("bbox").isNotNull() & bbox_ok)
        if dedup:
            df = df.dropDuplicates(["id"])
        return (
            df
            .withColumn("height", F.when(height_ok, F.col("height")))
            .withColumn("num_floors", F.when(floors_ok, F.col("num_floors")))
            .fillna({"subtype": "unknown", "class": "unknown"})
        )

    @staticmethod
    def region_expr(lon, lat):
        """Región aproximada (continente) a partir del centroide del bbox."""
        return (
            F.when(lon < -30, F.when(lat >= 15, "north_america").otherwise("latin_america"))
             .when(lon < 60,
                   F.when(lat >= 35, "europe")
                    .when((lat >= 12) & (lon >= 34), "middle_east")
                    .otherwise("africa"))
             .when((lat < -10) & (lon >= 110), "oceania")
             .otherwise("asia")
        )

    @staticmethod
    def derive_building_columns(df):
        """
        Column derivation:
          - lon/lat: centroide del bbox
          - region: continente aproximado (columna de partición)
          - primary_source: dataset de la primera fuente (sources[0].dataset)
          - source_count: número de fuentes del registro
          - has_name: el edificio tiene nombre primario
          - height_m: altura reportada o estimada con num_floors * 3 m
          - height_is_estimated: True si height_m viene de num_floors
          - height_category: low / mid / high / skyscraper / unknown
          - bbox_area_m2: área aproximada del bbox en m²
        """
        lon = (F.col("bbox.xmin") + F.col("bbox.xmax")) / 2
        lat = (F.col("bbox.ymin") + F.col("bbox.ymax")) / 2
        height_m = F.coalesce(F.col("height"), F.col("num_floors") * F.lit(SparkUtils.METERS_PER_FLOOR))

        return (
            df
            .withColumn("lon", lon.cast("double"))
            .withColumn("lat", lat.cast("double"))
            .withColumn("region", SparkUtils.region_expr(F.col("lon"), F.col("lat")))
            .withColumn("primary_source", F.coalesce(F.col("primary_source"), F.lit("unknown")))
            .withColumn("source_count", F.greatest(F.col("source_count"), F.lit(0)))
            .withColumn("has_name", F.col("name_primary").isNotNull())
            .withColumn("height_m", height_m.cast("double"))
            .withColumn("height_is_estimated", F.col("height").isNull() & F.col("height_m").isNotNull())
            .withColumn(
                "height_category",
                F.when(F.col("height_m").isNull(), "unknown")
                 .when(F.col("height_m") < 10, "low")
                 .when(F.col("height_m") < 30, "mid")
                 .when(F.col("height_m") < 100, "high")
                 .otherwise("skyscraper"))
            .withColumn(
                "bbox_area_m2",
                ((F.col("bbox.xmax") - F.col("bbox.xmin")) * 111320.0 * F.cos(F.radians(F.col("lat"))))
                * ((F.col("bbox.ymax") - F.col("bbox.ymin")) * 110540.0))
        )

    def create_source_lookup(self):
        """Tabla catálogo (pequeña) con información de cada dataset fuente de Overture."""
        data = [
            ("OpenStreetMap",                         "community",  "ODbL-1.0"),
            ("Microsoft ML Buildings",                "ml_derived", "ODbL-1.0"),
            ("Google Open Buildings",                 "ml_derived", "CC-BY-4.0 / ODbL-1.0"),
            ("Esri Community Maps",                   "authoritative", "CC-BY-4.0"),
            ("Instituto Geográfico Nacional (España)", "authoritative", "CC-BY-4.0"),
            ("USGS Lidar",                            "authoritative", "CC0-1.0"),
        ]
        schema = SparkUtils.generate_schema([
            ("dataset", "string"), ("source_type", "string"), ("source_license", "string"),
        ])
        return self._spark.createDataFrame(data, schema)

    @staticmethod
    def join_source_info(df, lookup):
        """
        Join: enriquece cada edificio con el tipo de fuente (community / ml_derived /
        authoritative). Left join para no perder edificios de fuentes no catalogadas;
        broadcast porque el catálogo es muy pequeño (evita el shuffle del lado grande).
        """
        return (
            df.join(F.broadcast(lookup), df["primary_source"] == lookup["dataset"], "left")
              .drop("dataset")
              .fillna({"source_type": "other", "source_license": "unknown"})
        )

    @staticmethod
    def aggregate_by_region_class(df):
        """Aggregation: métricas por región, clase y categoría de altura."""
        return (
            df.groupBy("region", "class", "height_category")
              .agg(
                  F.count("*").alias("building_count"),
                  F.round(F.avg("height_m"), 2).alias("avg_height_m"),
                  F.round(F.max("height_m"), 2).alias("max_height_m"),
                  F.round(F.avg("bbox_area_m2"), 2).alias("avg_bbox_area_m2"),
                  F.round(F.avg(F.col("has_name").cast("int")) * 100, 2).alias("pct_with_name"),
                  F.round(F.avg(F.col("height_m").isNotNull().cast("int")) * 100, 2).alias("pct_with_height"),
              )
        )

    @staticmethod
    def add_region_share(agg_df):
        """
        Join (agregado vs. totales): calcula qué porcentaje de los edificios de la
        región representa cada grupo, uniendo con el total por región.
        """
        totals = agg_df.groupBy("region").agg(F.sum("building_count").alias("region_total"))
        return (
            agg_df.join(totals, on="region", how="inner")
                  .withColumn("pct_of_region",
                              F.round(F.col("building_count") / F.col("region_total") * 100, 4))
        )

    @staticmethod
    def aggregate_by_source(df):
        """Aggregation: cobertura y calidad de datos por región y fuente."""
        return (
            df.groupBy("region", "source_type", "primary_source")
              .agg(
                  F.count("*").alias("building_count"),
                  F.round(F.avg(F.col("height_m").isNotNull().cast("int")) * 100, 2).alias("pct_with_height"),
                  F.round(F.avg(F.col("height_is_estimated").cast("int")) * 100, 2).alias("pct_height_estimated"),
                  F.round(F.avg("source_count"), 3).alias("avg_sources_per_building"),
              )
        )

    @staticmethod
    def filter_and_sort(df, min_buildings=1000):
        """Filtering & sorting: descarta grupos pequeños y ordena por región y tamaño."""
        return (
            df.filter(F.col("building_count") >= min_buildings)
              .orderBy(F.col("region").asc(), F.col("building_count").desc())
        )

    @staticmethod
    def select_detail_columns(df):
        """Columnas planas del detalle por edificio (sin structs/arrays) para persistir."""
        return df.select(
            "id", "region", "subtype", "class", "primary_source", "source_type",
            "source_count", "has_name", "height_m", "height_is_estimated",
            "height_category", "num_floors", "lon", "lat", "bbox_area_m2",
        )

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    @staticmethod
    def write_partitioned(df, path, partition_col, compact=True):
        """
        Escribe en Parquet particionado por partition_col.
        compact=True: repartition(partition_col) junta cada valor en una sola tarea,
        así cada carpeta tiene 1 archivo (ideal para tablas agregadas pequeñas).
        compact=False: para el detalle grande; cada tarea escribe su parte en paralelo.
        """
        if compact:
            df = df.repartition(partition_col)
        df.write.mode("overwrite").partitionBy(partition_col).parquet(path)
