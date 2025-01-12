import yaml
import os
from pyspark.sql import SparkSession
from analytics.analysis import run_analytics
from utils.log_config import setup_logging  

def main():
    # Initialize logging
    logger = setup_logging()

    # Load config file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    input_path = config["input_path"]
    output_path = config["output_path"]
    file_names = config["files"]
    
    # Initialize Spark Session
    spark = SparkSession.builder.appName("AnalyticsApp").getOrCreate()
    logger.info("Spark session created successfully.")
    
    # Run all analytics
    run_analytics(spark, input_path, file_names, output_path, logger)

    spark.stop()
    logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()
