import argparse

from pyspark import StorageLevel

from spark_utils_emr import SparkUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Overture buildings batch pipeline")
    parser.add_argument("--mode", choices=["debug", "pipeline"], default="pipeline")
    parser.add_argument("--input", required=True, help="Carpeta/archivo Parquet de entrada")
    parser.add_argument("--output", required=True,
                        help="debug: archivo .txt del reporte | pipeline: carpeta base de salida")
    parser.add_argument("--rows", type=int, default=20, help="Registros de muestra en los reportes")
    parser.add_argument("--count", action="store_true", help="(debug) contar todas las filas")
    parser.add_argument("--min-buildings", type=int, default=1000,
                        help="Mínimo de edificios para conservar un grupo agregado")
    parser.add_argument("--dedup", action="store_true",
                        help="Eliminar duplicados por id (shuffle completo, mucho más lento)")
    parser.add_argument("--write-detail", action="store_true",
                        help="También persistir el detalle por edificio (más pesado)")
    return parser.parse_args()


def run_debug(su, args):
    df = su.read_buildings(args.input)
    report = su.debug_report(df, title=f"Input: {args.input}", rows=args.rows, count_rows=args.count)
    su.write_text(args.output, report)
    print(f"Debug report written to {args.output}")


def run_pipeline(su, args):
    out = args.output.rstrip("/")

    # ---------------- Transformaciones (lazy) ----------------
    raw = su.read_buildings(args.input)
    buildings = su.select_building_columns(raw)
    clean = su.clean_buildings(buildings, dedup=args.dedup)
    enriched = su.derive_building_columns(clean)
    lookup = su.create_source_lookup()
    joined = su.join_source_info(enriched, lookup)
    # Se reutiliza en varias salidas: se marca para cache (también es lazy)
    joined = joined.persist(StorageLevel.MEMORY_AND_DISK)

    region_class = su.aggregate_by_region_class(joined)
    region_class = su.add_region_share(region_class)
    region_class = su.filter_and_sort(region_class, args.min_buildings)

    by_source = su.aggregate_by_source(joined)
    by_source = su.filter_and_sort(by_source, args.min_buildings)

    # ---------------- Persistencia (actions) ----------------
    su.write_partitioned(region_class, f"{out}/region_class_summary", "region")
    su.write_partitioned(by_source, f"{out}/source_summary", "region")
    if args.write_detail:
        su.write_partitioned(su.select_detail_columns(joined),
                             f"{out}/buildings_enriched", "region", compact=False)

    # ---------------- Validación: leer de vuelta lo escrito ----------------
    check = su.spark.read.parquet(f"{out}/region_class_summary")
    report = su.debug_report(check, title=f"Read back: {out}/region_class_summary",
                             rows=args.rows, count_rows=True)
    su.write_text(f"{out}/_reports/validation_region_class_summary.txt", report)
    joined.unpersist()
    print(f"Pipeline finished. Output in {out}")


def main():
    args = parse_args()
    su = SparkUtils("overture-buildings-pipeline")
    try:
        if args.mode == "debug":
            run_debug(su, args)
        else:
            run_pipeline(su, args)
    except Exception as e:
        # Deja un reporte en S3 para diagnosticar un step fallido
        target = args.output if args.mode == "debug" else f"{args.output.rstrip('/')}/_reports/error.txt"
        su.write_text(target, f"Mode: {args.mode}\nInput: {args.input}\n\n=== ERROR ===\n{type(e).__name__}: {e}\n")
        raise
    finally:
        su.spark.stop()


if __name__ == "__main__":
    main()
