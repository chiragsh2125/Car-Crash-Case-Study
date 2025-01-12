from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql import Window
import os

def run_analytics(spark, input_path, file_names, output_path, logger):
    """
    Run all analytics by loading datasets and processing them using PySpark DataFrame APIs.
    """
    # Load the datasets using spark.read.csv
    primary_person_df = spark.read.csv(
        os.path.join(input_path, file_names["primary_person"]), header=True, inferSchema=True
    )
    units_df = spark.read.csv(
        os.path.join(input_path, file_names["units"]), header=True, inferSchema=True 
    )
    damages_df = spark.read.csv(
        os.path.join(input_path, file_names["damages"]), header=True, inferSchema=True 
    )
    charges_df = spark.read.csv(
        os.path.join(input_path, file_names["charges"]), header=True, inferSchema=True 
    )

    # Run individual analytics
    analytics_1(units_df, primary_person_df, output_path, logger)
    analytics_2(units_df, output_path, logger)
    analytics_3(units_df, primary_person_df, output_path, logger)
    analytics_4(units_df, primary_person_df, output_path, logger)
    analytics_5( primary_person_df, output_path, logger)
    analytics_6(units_df, primary_person_df, output_path, logger)
    analytics_7(units_df, primary_person_df, output_path, logger)
    analytics_8(units_df, primary_person_df, output_path, logger)
    analytics_9(units_df, damages_df, output_path, logger)
    analytics_10(charges_df, primary_person_df, units_df, output_path, logger)

def analytics_1(units_df, primary_person_df, output_path, logger):
    """
    Find the number of crashes (accidents) in which the number of males killed is greater than 2.
    The DEATH_CNT is taken from units_df rather than primary_person_df.
    """

    # Step 1: Drop DEATH_CNT column from primary_person_df to avoid ambiguity
    primary_person_df = primary_person_df.drop("DEATH_CNT")

    # Step 2: Join the primary_person_df with units_df to fetch DEATH_CNT from units_df
    joined_df = primary_person_df.join(units_df, on="CRASH_ID", how="inner")

    # Step 3: Filter for males and those with DEATH_CNT > 2 from units_df
    result_df = joined_df.filter(
        (F.col("PRSN_GNDR_ID") == "MALE") & (F.col("DEATH_CNT") > 2)
    )

    # Step 4: Count the number of such crashes
    count = result_df.select("CRASH_ID").distinct().count()

    # Log the result
    logger.info(f"Analytics 1 result: {count} crashes where more than 2 males were killed.")

    # Step 5: Save the result to a file
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, "analytics_1.txt"), "w") as file:
        file.write(f"Number of crashes with more than 2 males killed: {count}\n")

def analytics_2(units_df, output_path, logger):
    """
    Find the number of two-wheelers booked for crashes.
    """
    result_df = units_df.filter(    
        col("UNIT_DESC_ID").like("%MOTOR VEHICLE%")
    )
    count = result_df.count()

    logger.info(f"Analytics 2 result: {count}")

    with open(os.path.join(output_path, "analytics_2.txt"), "w") as file:
        file.write(f"Number of two-wheelers involved in crashes: {count}\n")


def analytics_3(units_df, primary_person_df, output_path, logger):
    """
    Determine the Top 5 Vehicle Makes of the cars involved in crashes where the driver died and Airbags did not deploy.
    """
    joined_df = units_df.join(primary_person_df, 
                              (units_df["CRASH_ID"] == primary_person_df["CRASH_ID"]), "inner")

    result_df = joined_df.filter(
        (primary_person_df["DEATH_CNT"] > 0) & 
        (primary_person_df["PRSN_AIRBAG_ID"] == "NOT DEPLOYED") & 
        (units_df["VEH_MAKE_ID"].isNotNull())
    )

    make_count_df = result_df.groupBy("VEH_MAKE_ID").count()
    top_5_makes_df = make_count_df.orderBy(col("count").desc()).limit(5)

    top_5_makes = top_5_makes_df.collect()
    logger.info(f"Top 5 Vehicle Makes: {top_5_makes}")

    with open(os.path.join(output_path, "analytics_3.txt"), "w") as file:
        file.write("Top 5 Vehicle Makes (Driver Died & Airbags did not deploy):\n")
        for make in top_5_makes:
            file.write(f"{make['VEH_MAKE_ID']}: {make['count']}\n")


def analytics_4(units_df, primary_person_df, output_path, logger):
    """
    Determine the number of vehicles with drivers having valid licenses involved in hit-and-run accidents.
    """
    hit_and_run_df = units_df.filter(col("VEH_HNR_FL") == "Y")
    joined_df = hit_and_run_df.join(
        primary_person_df, 
        (hit_and_run_df["CRASH_ID"] == primary_person_df["CRASH_ID"])
    )
    
    valid_license_df = joined_df.filter(
        (col("DRVR_LIC_TYPE_ID").isin("DRIVER LICENSE", "COMMERCIAL DRIVER LIC.")) &  
        (~col("DRVR_LIC_CLS_ID").isin("UNLICENSED", "UNKNOWN"))
    )

    valid_license_vehicle_count = valid_license_df.select("VEH_MAKE_ID").distinct().count()
    logger.info(f"Number of Vehicles with Valid Licenses Involved in Hit-and-Run: {valid_license_vehicle_count}")

    with open(os.path.join(output_path, "analytics_4.txt"), "w") as file:
        file.write(f"Number of Vehicles with Valid Licenses Involved in Hit-and-Run: {valid_license_vehicle_count}\n")


def analytics_5(primary_person_df, output_path, logger):
    """
    Determine which state has the highest number of accidents in which females are not involved.
    """

    # Step 1: Filter for non-female records
    non_female_df = primary_person_df.filter(col("PRSN_GNDR_ID") != "FEMALE")

    # Step 2: Group by state and count the accidents
    state_accidents_df = non_female_df.groupBy("DRVR_LIC_STATE_ID").count()

    # Step 3: Find the state with the highest number of accidents
    highest_state_df = state_accidents_df.orderBy(col("count").desc()).limit(1)

    # Get the result
    highest_state = highest_state_df.collect()[0]
    logger.info(f"State with Highest Number of Accidents with No Female Involvement: {highest_state['DRVR_LIC_STATE_ID']}")

    # Step 4: Save the result to a file
    with open(os.path.join(output_path, "analytics_5.txt"), "w") as file:
        file.write(f"State with Highest Number of Accidents (No Female Involvement): {highest_state['DRVR_LIC_STATE_ID']}: {highest_state['count']}\n")


def analytics_6(units_df, primary_person_df, output_path, logger):
    """
    Determine the Top 3rd to 5th VEH_MAKE_IDs that contribute to the largest number of injuries, including death.
    """
    joined_df = units_df.alias('units').join(primary_person_df.alias('person'), 
                                              (F.col('units.CRASH_ID') == F.col('person.CRASH_ID')))
    
    make_injury_df = joined_df.groupBy("units.VEH_MAKE_ID").agg(
        (F.sum("person.TOT_INJRY_CNT") + F.sum("person.DEATH_CNT")).alias("total_injuries_and_deaths")
    )
    
    sorted_makes_df = make_injury_df.orderBy(F.col("total_injuries_and_deaths").desc())
    top_3rd_to_5th_makes_df = sorted_makes_df.limit(5).collect()[2:]

    logger.info("Top 3rd to 5th Vehicle Makes with Highest Number of Injuries (Including Death):")
    for make in top_3rd_to_5th_makes_df:
        logger.info(f"VEH_MAKE_ID: {make['VEH_MAKE_ID']} with {make['total_injuries_and_deaths']} total injuries and deaths")

    with open(os.path.join(output_path, "analytics_6.txt"), "w") as file:
        file.write("Top 3rd to 5th Vehicle Makes with Highest Number of Injuries (Including Death):\n")
        for make in top_3rd_to_5th_makes_df:
            file.write(f"{make['VEH_MAKE_ID']}: {make['total_injuries_and_deaths']}\n")


def analytics_7(units_df, primary_person_df, output_path, logger):
    """
    For each unique body style involved in crashes, mention the top ethnic user group.
    """
    joined_df = units_df.alias('units').join(primary_person_df.alias('person'),
                                              (F.col('units.CRASH_ID') == F.col('person.CRASH_ID')))
    
    ethnic_counts_df = joined_df.groupBy("units.VEH_BODY_STYL_ID", "person.PRSN_ETHNICITY_ID").agg(
        F.count("*").alias("ethnicity_count")
    )
    
    window_spec = Window.partitionBy("VEH_BODY_STYL_ID").orderBy(F.col("ethnicity_count").desc())
    top_ethnic_groups_df = ethnic_counts_df.withColumn("rank", F.row_number().over(window_spec)).filter(
        F.col("rank") == 1
    ).select("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID")

    ethnic_groups = top_ethnic_groups_df.collect()

    logger.info(f"Ethnic Group per Body Style with highest incidents:")
    for group in ethnic_groups:
        logger.info(f"Body Style {group['VEH_BODY_STYL_ID']}: Ethnicity {group['PRSN_ETHNICITY_ID']}")

    with open(os.path.join(output_path, "analytics_7.txt"), "w") as file:
        file.write("Top Ethnic Groups for Each Body Style:\n")
        for group in ethnic_groups:
            file.write(f"Body Style {group['VEH_BODY_STYL_ID']}: Ethnicity {group['PRSN_ETHNICITY_ID']}\n")

def analytics_8(units_df, primary_person_df, output_path, logger):
    """
    Analyze Top 5 Zip Codes with highest number of alcohol-related crashes (using Driver Zip Code).
    """
    # Join Unit and Primary Person DataFrames on CRASH_ID and UNIT_NBR
    joined_df = units_df.join(primary_person_df,
                              (units_df["CRASH_ID"] == primary_person_df["CRASH_ID"]))

    # Filter for alcohol as the contributing factor (in either CONGB_FACTR_1_ID or CONGB_FACTR_2_ID)
    alcohol_crashes_df = joined_df.filter(
        (F.col("CONTRIB_FACTR_1_ID").contains("ALCOHOL")) |
        (F.col("CONTRIB_FACTR_2_ID").contains("ALCOHOL"))
    )

    # Drop rows where DRVR_ZIP is missing
    alcohol_crashes_df = alcohol_crashes_df.filter(alcohol_crashes_df["DRVR_ZIP"].isNotNull())

    # Group by Driver Zip Code and count the number of occurrences
    zip_code_crashes_df = alcohol_crashes_df.groupBy("DRVR_ZIP").count()

    # Order by crash count descending and limit to top 5
    top_5_zip_codes_df = zip_code_crashes_df.orderBy(F.col("count").desc()).limit(5)

    # Collect results as a list
    top_5_zip_codes = top_5_zip_codes_df.collect()

    # Log the results
    logger.info(f"Top 5 Zip Codes with Highest Alcohol-Related Crashes: {top_5_zip_codes}")

    # Save the result to a file
    with open(os.path.join(output_path, "analytics_8.txt"), "w") as file:
        file.write("Top 5 Zip Codes with Highest Alcohol-Related Crashes:\n")
        for zip_code in top_5_zip_codes:
            file.write(f"Zip Code {zip_code['DRVR_ZIP']}: {zip_code['count']} crashes\n")      


def analytics_9(units_df, damages_df, output_path, logger):
    """
    Analyze and count distinct Crash IDs where:
      - No Damaged Property was observed
      - Damage Level (VEH_DMAG_SCL~) is above 4
      - Car avails Insurance

    Args:
        units_df (DataFrame): The Unit table DataFrame.
        damages_df (DataFrame): The Damages table DataFrame.
        output_path (str): The path to save the analysis result.
        logger (Logger): Logger for logging messages.
    """
    
    # Step 1: Filter damages table for no damaged property
    damages_filtered = damages_df.filter(
        (F.col("DAMAGED_PROPERTY").isNull()) | (F.col("DAMAGED_PROPERTY") == "NONE")
    )
    
    # Step 2: Filter units table for:
    # - Damage level > 4 on either VEH_DMAG_SCL_1_ID or VEH_DMAG_SCL_2_ID
    # - Exclude invalid damage types like "NA", "NO DAMAGE", "INVALID VALUE"
    # - Insurance is available (via FIN_RESP_TYPE_ID)
    units_filtered = units_df.filter(
        (
            (
                (units_df["VEH_DMAG_SCL_1_ID"] > "DAMAGED 4")
                & (~units_df["VEH_DMAG_SCL_1_ID"].isin(["NA", "NO DAMAGE", "INVALID VALUE"]))
            )
            | (
                (units_df["VEH_DMAG_SCL_2_ID"] > "DAMAGED 4")
                & (~units_df["VEH_DMAG_SCL_2_ID"].isin(["NA", "NO DAMAGE", "INVALID VALUE"]))
            )
        )
        & units_df["FIN_RESP_TYPE_ID"].isin(["PROOF OF LIABILITY INSURANCE", "LIABILITY INSURANCE POLICY"])
    )

    # Step 3: Join filtered damages and units datasets on CRASH_ID
    joined_df = units_filtered.join(damages_filtered, on="CRASH_ID", how="inner")

    # Step 4: Count distinct CRASH_IDs meeting the criteria
    distinct_crash_count = joined_df.select("CRASH_ID").distinct().count()

    # Log the result
    logger.info(f"Number of distinct Crash IDs meeting the criteria: {distinct_crash_count}")

    # Step 5: Save the result to a file
    result_file = os.path.join(output_path, "analytics_9.txt")
    with open(result_file, "w") as file:
        file.write(f"Number of distinct Crash IDs meeting the criteria: {distinct_crash_count}\n")

def analytics_10(charges_df, primary_person_df, units_df, output_path, logger):
    """
    Determine the Top 5 Vehicle Makes for drivers who are:
      - Charged with speeding-related offences
      - Licensed drivers
      - Using top 10 used vehicle colours
      - Cars licensed with the top 25 states with the highest number of offences
    """
    # Step 1: Get Top 25 States with highest number of offences
    top_25_states = [
        row[0] for row in (
            units_df
            .filter(F.col("VEH_LIC_STATE_ID").isNotNull())
            .groupBy("VEH_LIC_STATE_ID")
            .count()
            .orderBy(F.col("count").desc())
            .limit(25)
            .collect()
        )
    ]

    # Step 2: Get Top 10 used vehicle colors
    top_10_vehicle_colors = [
        row[0] for row in (
            units_df
            .filter(F.col("VEH_COLOR_ID").isNotNull() & (F.col("VEH_COLOR_ID") != "NA"))
            .groupBy("VEH_COLOR_ID")
            .count()
            .orderBy(F.col("count").desc())
            .limit(10)
            .collect()
        )
    ]

    # Step 3: Filter data based on speeding-related charges, licensed drivers, top colours, and top states
    filtered_data = (
        charges_df
        .filter(F.col("CHARGE").contains("SPEED"))
        .join(primary_person_df, on="CRASH_ID", how="inner")
        .filter(primary_person_df["DRVR_LIC_TYPE_ID"].isin(
            ["DRIVER LICENSE", "COMMERCIAL DRIVER LIC."]
        ))
        .join(units_df, on="CRASH_ID", how="inner")
        .filter(units_df["VEH_COLOR_ID"].isin(top_10_vehicle_colors))
        .filter(units_df["VEH_LIC_STATE_ID"].isin(top_25_states))
    )

    # Step 4: Identify the Top 5 Vehicle Makes
    top_5_vehicle_makes = (
        filtered_data
        .groupBy("VEH_MAKE_ID")
        .count()
        .orderBy(F.col("count").desc())
        .limit(5)
    )

    # Step 5: Collect results and log them
    top_5_vehicle_makes_list = top_5_vehicle_makes.collect()
    logger.info(f"Top 5 Vehicle Makes: {top_5_vehicle_makes_list}")

    # Step 6: Save the result to a file
    result_file = os.path.join(output_path, "analytics_10.txt")
    with open(result_file, "w") as file:
        file.write("Top 5 Vehicle Makes where drivers meet the specified criteria:\n")
        for vehicle in top_5_vehicle_makes_list:
            file.write(f"Vehicle Make {vehicle['VEH_MAKE_ID']}: {vehicle['count']} offences\n")

    return top_5_vehicle_makes_list
